#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: simple_networks.py
# --- Creation Date: 23-01-2020
# --- Last Modified: Thu 30 Jan 2020 15:11:24 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Simplified version of spatial-biased networks
with Factor-VAE-like implementation.
"""

import numpy as np
import pdb
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.ops.upfirdn_2d import upsample_2d, downsample_2d
from dnnlib.tflib.ops.upfirdn_2d import upsample_conv_2d, conv_downsample_2d
from dnnlib.tflib.ops.fused_bias_act import fused_bias_act
from training.networks_stylegan2 import get_weight, dense_layer, conv2d_layer
from training.networks_stylegan2 import apply_bias_act, naive_upsample_2d
from training.networks_stylegan2 import naive_downsample_2d, modulated_conv2d_layer
from training.networks_stylegan2 import minibatch_stddev_layer
from stn.stn import spatial_transformer_network as transformer
from training.spatial_biased_networks import G_mapping_spatial_biased_dsp

# NOTE: Do not import any application-specific modules here!
# Specify all network parameters as kwargs.


#----------------------------------------------------------------------------
# Spatial-Biased Generator
def G_main_simple_dsp(
        latents_in,  # First input: Latent vectors (Z) [minibatch, latent_size].
        labels_in,  # Second input: Conditioning labels [minibatch, label_size].
        is_training=False,  # Network is under training? Enables and disables specific features.
        is_validation=False,  # Network is under validation? Chooses which value to use for truncation_psi.
        return_dlatents=False,  # Return dlatents in addition to the images?
        is_template_graph=False,  # True = template graph constructed by the Network class, False = actual evaluation.
        components=dnnlib.EasyDict(
        ),  # Container for sub-networks. Retained between calls.
        mapping_func='G_mapping_spatial_biased_dsp',  # Build func name for the mapping network.
        synthesis_func='G_synthesis_simple_dsp',  # Build func name for the synthesis network.
        **kwargs):  # Arguments for sub-networks (mapping and synthesis).
    # Validate arguments.
    assert not is_training or not is_validation

    # Setup components.
    if 'synthesis' not in components:
        components.synthesis = tflib.Network(
            'G_spatial_biased_synthesis_dsp',
            func_name=globals()[synthesis_func],
            **kwargs)
    if 'mapping' not in components:
        components.mapping = tflib.Network('G_spatial_biased_mapping_dsp',
                                           func_name=globals()[mapping_func],
                                           dlatent_broadcast=None,
                                           **kwargs)

    # Evaluate mapping network.
    dlatents = components.mapping.get_output_for(latents_in,
                                                 labels_in,
                                                 is_training=is_training,
                                                 **kwargs)
    dlatents = tf.cast(dlatents, tf.float32)

    # Evaluate synthesis network.
    images_out = components.synthesis.get_output_for(
        dlatents,
        is_training=is_training,
        force_clean_graph=is_template_graph,
        **kwargs)

    # Return requested outputs.
    images_out = tf.identity(images_out, name='images_out')
    if return_dlatents:
        return images_out, dlatents
    return images_out


#----------------------------------------------------------------------------
# StyleGAN2-like spatial-biased simplified synthesis network for dsprites.


def G_synthesis_simple_dsp(
        dlatents_in,  # Input: Disentangled latents (W) [minibatch, dlatent_size].
        dlatent_size=7,  # Disentangled latent (W) dimensionality. Including discrete info, rotation, scaling, and xy translation.
        D_global_size=3,  # Discrete latents.
        sb_C_global_size=4,  # Continuous latents.
        label_size=0,  # Label dimensionality, 0 if no labels.
        num_channels=1,  # Number of output color channels.
        nonlinearity='relu',  # Activation function: 'relu', 'lrelu', etc.
        dtype='float32',  # Data type to use for activations and outputs.
        resample_kernel=[
            1, 3, 3, 1
        ],  # Low-pass filter to apply when resampling activations. None = no filtering.
        fused_modconv=True,  # Implement modulated_conv2d_layer() as a single fused op?
        **_kwargs):  # Ignore unrecognized keyword args.

    act = nonlinearity
    images_out = None

    # Primary inputs.
    assert dlatent_size == D_global_size + sb_C_global_size
    n_cat = label_size + D_global_size
    dlatents_in.set_shape([None, label_size + dlatent_size])
    dlatents_in = tf.cast(dlatents_in, dtype)

    # Return rotation matrix
    def get_r_matrix(r_latents, cond_latent):
        # r_latents: [-2., 2.] -> [0, 2*pi]
        with tf.variable_scope('Condition0'):
            cond = apply_bias_act(dense_layer(cond_latent, fmaps=128), act=act)
        with tf.variable_scope('Condition1'):
            cond = apply_bias_act(dense_layer(cond, fmaps=1), act='sigmoid')
        rad = (r_latents + 2) / 4. * 2. * np.pi
        rad = rad * cond
        tt_00 = tf.math.cos(rad)
        tt_01 = -tf.math.sin(rad)
        tt_02 = tf.zeros_like(rad)
        tt_10 = tf.math.sin(rad)
        tt_11 = tf.math.cos(rad)
        tt_12 = tf.zeros_like(rad)
        theta = tf.concat([tt_00, tt_01, tt_02, tt_10, tt_11, tt_12], axis=1)
        return theta

    # Return scaling matrix
    def get_s_matrix(s_latents, cond_latent):
        # s_latents: [-2., 2.] -> [1, 3]
        with tf.variable_scope('Condition0'):
            cond = apply_bias_act(dense_layer(cond_latent, fmaps=128), act=act)
        with tf.variable_scope('Condition1'):
            cond = apply_bias_act(dense_layer(cond, fmaps=1), act='sigmoid')
        scale = (s_latents / 2. + 2.) * cond
        tt_00 = scale
        tt_01 = tf.zeros_like(scale)
        tt_02 = tf.zeros_like(scale)
        tt_10 = tf.zeros_like(scale)
        tt_11 = scale
        tt_12 = tf.zeros_like(scale)
        theta = tf.concat([tt_00, tt_01, tt_02, tt_10, tt_11, tt_12], axis=1)
        return theta

    # Return translation matrix
    def get_t_matrix(t_latents, cond_latent):
        # t_latents[:, 0]: [-2., 2.] -> [-0.5, 0.5]
        # t_latents[:, 1]: [-2., 2.] -> [-0.5, 0.5]
        with tf.variable_scope('Condition0'):
            cond = apply_bias_act(dense_layer(cond_latent, fmaps=128), act=act)
        with tf.variable_scope('Condition1'):
            cond = apply_bias_act(dense_layer(cond, fmaps=1), act='sigmoid')
        xy_shift = t_latents / 4. * cond
        tt_00 = tf.ones_like(xy_shift[:, 0:1])
        tt_01 = tf.zeros_like(xy_shift[:, 0:1])
        tt_02 = xy_shift[:, 0:1]
        tt_10 = tf.zeros_like(xy_shift[:, 1:])
        tt_11 = tf.ones_like(xy_shift[:, 1:])
        tt_12 = xy_shift[:, 1:]
        theta = tf.concat([tt_00, tt_01, tt_02, tt_10, tt_11, tt_12], axis=1)
        return theta

    # Apply spatial transform
    def apply_st(x, st_matrix):
        with tf.variable_scope('Transform'):
            x = tf.transpose(x, [0, 2, 3, 1])  # NCHW -> NHWC
            x = transformer(x, st_matrix, out_dims=x.shape.as_list()[1:3])
            x = tf.transpose(x, [0, 3, 1, 2])  # NHWC -> NCHW
        return x

    def torgb(x, y):
        with tf.variable_scope('ToRGB'):
            t = apply_bias_act(conv2d_layer(x, fmaps=num_channels, kernel=1))
            return t if y is None else y + t

    # Early layers.
    y = None
    with tf.variable_scope('4x4'):
        with tf.variable_scope('Const'):
            x = tf.get_variable('const',
                                shape=[1, 64, 4, 4],
                                initializer=tf.initializers.random_normal())
            x = tf.tile(tf.cast(x, dtype), [tf.shape(dlatents_in)[0], 1, 1, 1])

        with tf.variable_scope('4x4Conv'):
            w = get_weight([3, 3, x.shape[1].value, 64])
            x = tf.nn.conv2d(x,
                             tf.cast(w, x.dtype),
                             data_format='NCHW',
                             strides=[1, 1, 1, 1],
                             padding='SAME')
            x = apply_bias_act(x, act=act)

    with tf.variable_scope('8x8ModulatedConv'):
        x = apply_bias_act(modulated_conv2d_layer(
            x,
            dlatents_in[:, :n_cat],
            fmaps=64,
            kernel=3,
            up=True,
            resample_kernel=resample_kernel,
            fused_modconv=fused_modconv),
                           act=act)

    with tf.variable_scope('16x16'):
        w = get_weight([4, 4, x.shape[1].value, 32])
        # Transpose weights.
        w = tf.transpose(w, [0, 1, 3, 2])
        x = tf.nn.conv2d_transpose(
            x,
            w,
            output_shape=[tf.shape(dlatents_in)[0], 32, 16, 16],
            strides=[1, 1, 2, 2],
            padding='SAME',
            data_format='NCHW')
        x = apply_bias_act(x, act=act)

        with tf.variable_scope('rotation'):
            r_matrix = get_r_matrix(dlatents_in[:, n_cat:n_cat + 1],
                                    dlatents_in[:, :n_cat])
            x = apply_st(x, r_matrix)
        with tf.variable_scope('scale'):
            s_matrix = get_s_matrix(dlatents_in[:, n_cat + 1:n_cat + 2],
                                    dlatents_in[:, :n_cat])
            x = apply_st(x, s_matrix)
        with tf.variable_scope('translation'):
            t_matrix = get_t_matrix(dlatents_in[:, n_cat + 2:],
                                    dlatents_in[:, :n_cat])
            x = apply_st(x, t_matrix)

    with tf.variable_scope('32x32'):
        w = get_weight([4, 4, x.shape[1].value, 32])
        # Transpose weights.
        w = tf.transpose(w, [0, 1, 3, 2])
        x = tf.nn.conv2d_transpose(
            x,
            w,
            output_shape=[tf.shape(dlatents_in)[0], 32, 32, 32],
            strides=[1, 1, 2, 2],
            padding='SAME',
            data_format='NCHW')
        x = apply_bias_act(x, act=act)

    with tf.variable_scope('64x64'):
        w = get_weight([4, 4, x.shape[1].value, 1])
        # Transpose weights.
        w = tf.transpose(w, [0, 1, 3, 2])
        x = tf.nn.conv2d_transpose(
            x,
            w,
            output_shape=[tf.shape(dlatents_in)[0], 1, 64, 64],
            strides=[1, 1, 2, 2],
            padding='SAME',
            data_format='NCHW')
        x = apply_bias_act(x)

    images_out = x
    assert images_out.dtype == tf.as_dtype(dtype)
    return tf.identity(images_out, name='images_out')


#----------------------------------------------------------------------------
# Simple discriminator for dsprites dataset usage.
# This is similar to encoder in FactorVAE.


def D_simple_dsp(
        images_in,  # First input: Images [minibatch, channel, height, width].
        labels_in,  # Second input: Labels [minibatch, label_size].
        num_channels=1,  # Number of input color channels. Overridden based on dataset.
        label_size=0,  # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
        nonlinearity='relu',  # Activation function: 'relu', 'lrelu', etc.
        dtype='float32',  # Data type to use for activations and outputs.
        **_kwargs):  # Ignore unrecognized keyword args.

    act = nonlinearity

    images_in.set_shape([None, num_channels, 64, 64])
    labels_in.set_shape([None, label_size])
    images_in = tf.cast(images_in, dtype)
    labels_in = tf.cast(labels_in, dtype)

    x = images_in
    with tf.variable_scope('32x32'):
        w = get_weight([4, 4, x.shape[1].value, 32])
        x = tf.nn.conv2d(x,
                         tf.cast(w, x.dtype),
                         data_format='NCHW',
                         strides=[1, 1, 2, 2],
                         padding='SAME')
        x = apply_bias_act(x, act=act)

    with tf.variable_scope('16x16'):
        w = get_weight([4, 4, x.shape[1].value, 32])
        x = tf.nn.conv2d(x,
                         tf.cast(w, x.dtype),
                         data_format='NCHW',
                         strides=[1, 1, 2, 2],
                         padding='SAME')
        x = apply_bias_act(x, act=act)

    with tf.variable_scope('8x8'):
        w = get_weight([4, 4, x.shape[1].value, 64])
        x = tf.nn.conv2d(x,
                         tf.cast(w, x.dtype),
                         data_format='NCHW',
                         strides=[1, 1, 2, 2],
                         padding='SAME')
        x = apply_bias_act(x, act=act)

    with tf.variable_scope('4x4'):
        w = get_weight([4, 4, x.shape[1].value, 64])
        x = tf.nn.conv2d(x,
                         tf.cast(w, x.dtype),
                         data_format='NCHW',
                         strides=[1, 1, 2, 2],
                         padding='SAME')
        x = apply_bias_act(x, act=act)

    with tf.variable_scope('output'):
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
        w = get_weight([x.shape[1].value, 128])
        w = tf.cast(w, x.dtype)
        x = tf.matmul(x, w)
        x = apply_bias_act(x, act=act)

    # Output layer with label conditioning from "Which Training Methods for GANs do actually Converge?"
    with tf.variable_scope('Score'):
        w = get_weight([x.shape[1].value, 1])
        w = tf.cast(w, x.dtype)
        x = tf.matmul(x, w)
        x = apply_bias_act(x)
    scores_out = x

    # Output.
    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')
    return scores_out


#----------------------------------------------------------------------------
# StyleGAN2-like spatial-biased synthesis network for dsprites.
# Spatial modifications are done in a single layer.


def G_synthesis_sb_singlelayer_modi_dsp(
        dlatents_in,  # Input: Disentangled latents (W) [minibatch, dlatent_size].
        dlatent_size=7,  # Disentangled latent (W) dimensionality. Including discrete info, rotation, scaling, and xy translation.
        D_global_size=3,  # Discrete latents.
        sb_C_global_size=4,  # Continuous latents.
        label_size=0,  # Label dimensionality, 0 if no labels.
        num_channels=1,  # Number of output color channels.
        resolution=64,  # Output resolution.
        fmap_base=16 <<
        10,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_min=1,  # Minimum number of feature maps in any layer.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        architecture='skip',  # Architecture: 'orig', 'skip', 'resnet'.
        nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        dtype='float32',  # Data type to use for activations and outputs.
        resample_kernel=[
            1, 3, 3, 1
        ],  # Low-pass filter to apply when resampling activations. None = no filtering.
        fused_modconv=True,  # Implement modulated_conv2d_layer() as a single fused op?
        **_kwargs):  # Ignore unrecognized keyword args.
    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4

    def nf(stage):
        return np.clip(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_min,
                       fmap_max)

    assert architecture in ['orig', 'skip', 'resnet']
    act = nonlinearity
    images_out = None

    # Primary inputs.
    assert dlatent_size == D_global_size + sb_C_global_size
    n_cat = label_size + D_global_size
    dlatents_in.set_shape([None, label_size + dlatent_size])
    dlatents_in = tf.cast(dlatents_in, dtype)

    # Return rotation matrix
    def get_r_matrix(r_latents, cond_latent):
        # r_latents: [-2., 2.] -> [0, 2*pi]
        with tf.variable_scope('Condition0'):
            cond = apply_bias_act(dense_layer(cond_latent, fmaps=128), act=act)
        with tf.variable_scope('Condition1'):
            cond = apply_bias_act(dense_layer(cond, fmaps=1), act='sigmoid')
        rad = (r_latents + 2) / 4. * 2. * np.pi
        rad = rad * cond
        tt_00 = tf.math.cos(rad)
        tt_01 = -tf.math.sin(rad)
        tt_02 = tf.zeros_like(rad)
        tt_10 = tf.math.sin(rad)
        tt_11 = tf.math.cos(rad)
        tt_12 = tf.zeros_like(rad)
        theta = tf.concat([tt_00, tt_01, tt_02, tt_10, tt_11, tt_12], axis=1)
        return theta

    # Return scaling matrix
    def get_s_matrix(s_latents, cond_latent):
        # s_latents: [-2., 2.] -> [1, 3]
        with tf.variable_scope('Condition0'):
            cond = apply_bias_act(dense_layer(cond_latent, fmaps=128), act=act)
        with tf.variable_scope('Condition1'):
            cond = apply_bias_act(dense_layer(cond, fmaps=1), act='sigmoid')
        scale = (s_latents / 2. + 2.) * cond
        tt_00 = scale
        tt_01 = tf.zeros_like(scale)
        tt_02 = tf.zeros_like(scale)
        tt_10 = tf.zeros_like(scale)
        tt_11 = scale
        tt_12 = tf.zeros_like(scale)
        theta = tf.concat([tt_00, tt_01, tt_02, tt_10, tt_11, tt_12], axis=1)
        return theta

    # Return shear matrix
    def get_sh_matrix(sh_latents, cond_latent):
        # sh_latents[:, 0]: [-2., 2.] -> [-1., 1.]
        # sh_latents[:, 1]: [-2., 2.] -> [-1., 1.]
        with tf.variable_scope('Condition0x'):
            cond_x = apply_bias_act(dense_layer(cond_latent, fmaps=128),
                                    act=act)
        with tf.variable_scope('Condition1x'):
            cond_x = apply_bias_act(dense_layer(cond_x, fmaps=1),
                                    act='sigmoid')
        with tf.variable_scope('Condition0y'):
            cond_y = apply_bias_act(dense_layer(cond_latent, fmaps=128),
                                    act=act)
        with tf.variable_scope('Condition1y'):
            cond_y = apply_bias_act(dense_layer(cond_y, fmaps=1),
                                    act='sigmoid')
        cond = tf.concat([cond_x, cond_y], axis=1)
        xy_shear = sh_latents / 2. * cond
        tt_00 = tf.ones_like(xy_shear[:, 0:1])
        tt_01 = xy_shear[:, 0:1]
        tt_02 = tf.zeros_like(xy_shear[:, 0:1])
        tt_10 = xy_shear[:, 1:]
        tt_11 = tf.ones_like(xy_shear[:, 1:])
        tt_12 = tf.zeros_like(xy_shear[:, 1:])
        theta = tf.concat([tt_00, tt_01, tt_02, tt_10, tt_11, tt_12], axis=1)
        return theta

    # Return translation matrix
    def get_t_matrix(t_latents, cond_latent):
        # t_latents[:, 0]: [-2., 2.] -> [-0.5, 0.5]
        # t_latents[:, 1]: [-2., 2.] -> [-0.5, 0.5]
        with tf.variable_scope('Condition0x'):
            cond_x = apply_bias_act(dense_layer(cond_latent, fmaps=128),
                                    act=act)
        with tf.variable_scope('Condition1x'):
            cond_x = apply_bias_act(dense_layer(cond_x, fmaps=1),
                                    act='sigmoid')
        with tf.variable_scope('Condition0y'):
            cond_y = apply_bias_act(dense_layer(cond_latent, fmaps=128),
                                    act=act)
        with tf.variable_scope('Condition1y'):
            cond_y = apply_bias_act(dense_layer(cond_y, fmaps=1),
                                    act='sigmoid')
        cond = tf.concat([cond_x, cond_y], axis=1)
        xy_shift = t_latents / 4. * cond
        tt_00 = tf.ones_like(xy_shift[:, 0:1])
        tt_01 = tf.zeros_like(xy_shift[:, 0:1])
        tt_02 = xy_shift[:, 0:1]
        tt_10 = tf.zeros_like(xy_shift[:, 1:])
        tt_11 = tf.ones_like(xy_shift[:, 1:])
        tt_12 = xy_shift[:, 1:]
        theta = tf.concat([tt_00, tt_01, tt_02, tt_10, tt_11, tt_12], axis=1)
        return theta


    # # Return translation matrix. x, y share parameters.
    # def get_t_matrix(t_latents, cond_latent):
        # # t_latents[:, 0]: [-2., 2.] -> [-0.5, 0.5]
        # # t_latents[:, 1]: [-2., 2.] -> [-0.5, 0.5]
        # with tf.variable_scope('Condition0'):
            # cond = apply_bias_act(dense_layer(cond_latent, fmaps=128), act=act)
        # with tf.variable_scope('Condition1'):
            # cond = apply_bias_act(dense_layer(cond, fmaps=1), act='sigmoid')
        # xy_shift = t_latents / 4. * cond
        # tt_00 = tf.ones_like(xy_shift[:, 0:1])
        # tt_01 = tf.zeros_like(xy_shift[:, 0:1])
        # tt_02 = xy_shift[:, 0:1]
        # tt_10 = tf.zeros_like(xy_shift[:, 1:])
        # tt_11 = tf.ones_like(xy_shift[:, 1:])
        # tt_12 = xy_shift[:, 1:]
        # theta = tf.concat([tt_00, tt_01, tt_02, tt_10, tt_11, tt_12], axis=1)
        # return theta

    # Apply spatial transform
    def apply_st(x, st_matrix, idx):  # idx: 2, 3, 4
        with tf.variable_scope('Transform'):
            x = tf.transpose(x, [0, 2, 3, 1])  # NCHW -> NHWC
            x = transformer(x, st_matrix, out_dims=x.shape.as_list()[1:3])
            x = tf.transpose(x, [0, 3, 1, 2])  # NHWC -> NCHW
        return x

    def upsample(y):
        with tf.variable_scope('Upsample'):
            return upsample_2d(y, k=resample_kernel)

    def torgb(x, y):
        with tf.variable_scope('ToRGB'):
            t = apply_bias_act(conv2d_layer(x, fmaps=num_channels, kernel=1))
            return t if y is None else y + t

    # Early layers.
    y = None
    with tf.variable_scope('4x4'):
        with tf.variable_scope('Const'):
            x = tf.get_variable('const',
                                shape=[1, nf(1), 4, 4],
                                initializer=tf.initializers.random_normal())
            x = tf.tile(tf.cast(x, dtype), [tf.shape(dlatents_in)[0], 1, 1, 1])
        with tf.variable_scope('Upconv8x8'):
            x = apply_bias_act(conv2d_layer(x,
                                            fmaps=nf(1),
                                            kernel=3,
                                            up=True,
                                            resample_kernel=resample_kernel),
                               act=act)
        with tf.variable_scope('Conv0'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(1), kernel=3), act=act)
        with tf.variable_scope('ModulatedConv'):
            x = apply_bias_act(modulated_conv2d_layer(
                x,
                dlatents_in[:, :n_cat],
                fmaps=nf(2),
                kernel=3,
                up=False,
                resample_kernel=resample_kernel,
                fused_modconv=fused_modconv),
                               act=act)

        with tf.variable_scope('Conv1'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(2), kernel=3), act=act)

    # Rotation layers.
    with tf.variable_scope('16x16'):
        with tf.variable_scope('Upconv'):
            x = apply_bias_act(conv2d_layer(x,
                                            fmaps=nf(2),
                                            kernel=3,
                                            up=True,
                                            resample_kernel=resample_kernel),
                               act=act)
        with tf.variable_scope('Conv0'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(1), kernel=3), act=act)
        with tf.variable_scope('rotation'):
            r_matrix = get_r_matrix(dlatents_in[:, n_cat:n_cat + 1],
                                    dlatents_in[:, :n_cat])
            x = apply_st(x, r_matrix, 2)
        with tf.variable_scope('scale'):
            s_matrix = get_s_matrix(dlatents_in[:, n_cat + 1:n_cat + 2],
                                    dlatents_in[:, :n_cat])
            x = apply_st(x, s_matrix, 3)
        if sb_C_global_size == 6:
            with tf.variable_scope('shear'):
                sh_matrix = get_sh_matrix(dlatents_in[:, n_cat + 2:n_cat + 4],
                                          dlatents_in[:, :n_cat])
                x = apply_st(x, sh_matrix, 3)
            with tf.variable_scope('translation'):
                t_matrix = get_t_matrix(dlatents_in[:, n_cat + 4:],
                                        dlatents_in[:, :n_cat])
                x = apply_st(x, t_matrix, 3)
        else:
            with tf.variable_scope('translation'):
                t_matrix = get_t_matrix(dlatents_in[:, n_cat + 2:],
                                        dlatents_in[:, :n_cat])
                x = apply_st(x, t_matrix, 3)
        with tf.variable_scope('Conv1'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(2), kernel=3), act=act)

    with tf.variable_scope('32x32'):
        with tf.variable_scope('UpConv'):
            x = apply_bias_act(conv2d_layer(x,
                                            fmaps=nf(3),
                                            kernel=3,
                                            up=True,
                                            resample_kernel=resample_kernel),
                               act=act)
        with tf.variable_scope('Conv'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(3), kernel=3), act=act)
    with tf.variable_scope('64x64'):
        with tf.variable_scope('UpConv'):
            x = apply_bias_act(conv2d_layer(x,
                                            fmaps=nf(4),
                                            kernel=3,
                                            up=True,
                                            resample_kernel=resample_kernel),
                               act=act)
        with tf.variable_scope('Conv'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(4), kernel=3), act=act)

    y = torgb(x, y)

    # # Tail layers.
    # for res in range(6, resolution_log2 + 1):
    # with tf.variable_scope('%dx%d' % (res * 2, res * 2)):
    # x = apply_bias_act(conv2d_layer(x,
    # fmaps=nf(res),
    # kernel=1,
    # up=True,
    # resample_kernel=resample_kernel),
    # act=act)
    # if architecture == 'skip':
    # y = upsample(y)
    # if architecture == 'skip' or res == resolution_log2:
    # y = torgb(x, y)
    images_out = y
    assert images_out.dtype == tf.as_dtype(dtype)
    return tf.identity(images_out, name='images_out')
