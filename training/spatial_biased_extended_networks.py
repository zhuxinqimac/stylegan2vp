#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: spatial_biased_extended_networks.py
# --- Creation Date: 28-01-2020
# --- Last Modified: Sun 02 Feb 2020 16:12:50 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Other spatial biased networks.
Usually synthesis networks.
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

#----------------------------------------------------------------------------
# StyleGAN2-like general biased synthesis network for 2D datasets.
# This model includes:
# 0) global Labels (at top);
# 1) global D_latents (at top);
# 2) global C_latents (at top);
# 3) global spatial-biased C_latents (at middle, include rotation, scaling, xy shearing, and xy translation);
# 4) local heatmap*features learned C_latents (at top and bottom);
# 5) local heatmap learned C_latents (at top and bottom);
# 6) noise (at top and middle and bottom).


def G_synthesis_sb_general_dsp(
        dlatents_withl_in,  # Input: Disentangled latents (W) [minibatch, label_size+dlatent_size].
        dlatent_size=7,  # Disentangled latent (W) dimensionality. Including discrete info, rotation, scaling, xy shearing, and xy translation.
        label_size=0,  # Label dimensionality, 0 if no labels.
        D_global_size=3,  # Global D_latents.
        C_global_size=0,  # Global C_latents.
        sb_C_global_size=4,  # Global spatial-biased C_latents.
        C_local_hfeat_size=0,  # Local heatmap*features learned C_latents.
        C_local_heat_size=0,  # Local heatmap learned C_latents.
        num_channels=1,  # Number of output color channels.
        resolution=64,  # Output resolution.
        nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        dtype='float32',  # Data type to use for activations and outputs.
        resample_kernel=[
            1, 3, 3, 1
        ],  # Low-pass filter to apply when resampling activations. None = no filtering.
        fused_modconv=True,  # Implement modulated_conv2d_layer() as a single fused op?
        use_noise=False,
        randomize_noise=True,  # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
        **_kwargs):  # Ignore unrecognized keyword args.
    '''
    dlatents_withl_in: dims contain: [label, D_global, C_global, sb_C_global,
                                C_local_hfeat, C_local_feat]
    '''
    resolution_log2 = int(np.log2(resolution))  # == 6 for resolution 64
    assert resolution == 2**resolution_log2 and resolution >= 4
    num_layers = resolution_log2 * 2 - 2  # == 10 for resolution 64

    act = nonlinearity
    images_out = None

    # Primary inputs.
    assert dlatent_size == D_global_size + C_global_size + sb_C_global_size + \
        C_local_hfeat_size + C_local_heat_size
    n_cat = label_size + D_global_size
    dlatents_withl_in.set_shape([None, label_size + dlatent_size])
    dlatents_withl_in = tf.cast(dlatents_withl_in, dtype)
    n_content = label_size + D_global_size + C_global_size

    # Noise inputs.
    noise_inputs = []
    for layer_idx in range(num_layers - 3):
        res = (layer_idx + 7) // 2  # [3, 4, 4, 5, 5, 6, 6]
        shape = [1, 1, 2**res, 2**res]
        noise_inputs.append(
            tf.get_variable('noise%d' % layer_idx,
                            shape=shape,
                            initializer=tf.initializers.random_normal(),
                            trainable=False))

    # Single convolution layer with all the bells and whistles.
    def noised_conv_layer(x, layer_idx, fmaps, kernel, up=False):
        x = conv2d_layer(x,
                         fmaps=fmaps,
                         up=up,
                         kernel=kernel,
                         resample_kernel=resample_kernel)
        if use_noise:
            if randomize_noise:
                noise = tf.random_normal(
                    [tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
            else:
                noise = tf.cast(noise_inputs[layer_idx], x.dtype)
            noise_strength = tf.get_variable(
                'noise_strength',
                shape=[],
                initializer=tf.initializers.zeros())
            x += noise * tf.cast(noise_strength, x.dtype)
        return apply_bias_act(x, act=act)

    # Early layers consists of 4x4 constant layer,
    # label+global discrete latents,
    # and global continuous latents.
    y = None
    with tf.variable_scope('4x4'):
        with tf.variable_scope('Const'):
            x = tf.get_variable('const',
                                shape=[1, 128, 4, 4],
                                initializer=tf.initializers.random_normal())
            x = tf.tile(tf.cast(x, dtype),
                        [tf.shape(dlatents_withl_in)[0], 1, 1, 1])
        with tf.variable_scope('Upconv'):
            x = apply_bias_act(conv2d_layer(x,
                                            fmaps=128,
                                            kernel=3,
                                            up=True,
                                            resample_kernel=resample_kernel),
                               act=act)

    with tf.variable_scope('8x8'):
        with tf.variable_scope('Conv0'):
            x = apply_bias_act(conv2d_layer(x, fmaps=128, kernel=3), act=act)
        with tf.variable_scope('Label_Dglobal_control'):
            x = apply_bias_act(modulated_conv2d_layer(
                x,
                dlatents_withl_in[:, :n_cat],
                fmaps=128,
                kernel=3,
                up=False,
                resample_kernel=resample_kernel,
                fused_modconv=fused_modconv),
                               act=act)
        with tf.variable_scope('After_DiscreteGlobal_noised'):
            x = noised_conv_layer(x, layer_idx=0, fmaps=128, kernel=3)
        with tf.variable_scope('Cglobal_control'):
            start_idx = n_cat
            x = apply_bias_act(modulated_conv2d_layer(
                x,
                dlatents_withl_in[:, start_idx:start_idx + C_global_size],
                fmaps=128,
                kernel=3,
                up=False,
                resample_kernel=resample_kernel,
                fused_modconv=fused_modconv),
                               act=act)
        with tf.variable_scope('After_ContinuousGlobal_noised'):
            x = noised_conv_layer(x, layer_idx=1, up=True, fmaps=128, kernel=3)

    # Spatial biased layers.
    with tf.variable_scope('16x16'):
        if C_local_hfeat_size > 0:
            with tf.variable_scope('LocalHFeat_C_latents'):
                with tf.variable_scope('ConstFeats'):
                    const_feats = tf.get_variable(
                        'constfeats',
                        shape=[1, C_local_hfeat_size, 32, 1, 1],
                        initializer=tf.initializers.random_normal())
                    const_feats = tf.tile(
                        tf.cast(const_feats,
                                dtype), [tf.shape(const_feats)[0], 1, 1, 1, 1])
                with tf.variable_scope('ControlAttHeat'):
                    hfeat_start_idx = label_size + D_global_size + C_global_size + \
                        sb_C_global_size
                    att_heat = get_att_heat(x,
                                            nheat=C_local_hfeat_size,
                                            act=act)
                    att_heat = tf.reshape(
                        att_heat,
                        [tf.shape(att_heat)[0], C_local_hfeat_size, 1] +
                        att_heat.shape.as_list()[2:4])
                    # C_local_heat latent [-2, 2] --> [0, 1]
                    hfeat_modifier = (2 + dlatents_withl_in[:, hfeat_start_idx:hfeat_start_idx + \
                                                     C_local_hfeat_size]) / 4.
                    hfeat_modifier = get_conditional_modifier(
                        hfeat_modifier,
                        dlatents_withl_in[:, :n_content],
                        act=act)
                    hfeat_modifier = tf.reshape(
                        hfeat_modifier,
                        [tf.shape(x)[0], C_local_hfeat_size, 1, 1, 1])
                    att_heat = att_heat * hfeat_modifier
                    added_feats = const_feats * att_heat
                    added_feats = tf.reshape(added_feats, [
                        tf.shape(att_heat)[0],
                        C_local_hfeat_size * att_heat.shape.as_list()[2]
                    ] + att_heat.shape.as_list()[3:5])
                    x = tf.concat([x, added_feats], axis=1)

        with tf.variable_scope('SpatialBiased_C_global'):
            # Rotation layers.
            start_idx = start_idx + C_global_size
            with tf.variable_scope('Rotation'):
                r_matrix = get_r_matrix(
                    dlatents_withl_in[:, start_idx:start_idx + 1],
                    dlatents_withl_in[:, :n_content],
                    act=act)
                x = apply_st(x, r_matrix, up=False, fmaps=128, act=act)
            with tf.variable_scope('After_Rotation_noised'):
                x = noised_conv_layer(x, layer_idx=2, fmaps=128, kernel=3)
            # Scaling layers.
            start_idx = start_idx + 1
            with tf.variable_scope('Scaling'):
                s_matrix = get_s_matrix(
                    dlatents_withl_in[:, start_idx:start_idx + 1],
                    dlatents_withl_in[:, :n_content],
                    act=act)
                x = apply_st(x, s_matrix, up=False, fmaps=128, act=act)
            with tf.variable_scope('After_Scaling_noised'):
                x = noised_conv_layer(x,
                                      layer_idx=3,
                                      up=True,
                                      fmaps=128,
                                      kernel=3)

    with tf.variable_scope('32x32'):
        with tf.variable_scope('SpatialBiased_C_global'):
            # Shearing layers.
            with tf.variable_scope('Shearing'):
                start_idx = start_idx + 1
                sh_matrix = get_sh_matrix(
                    dlatents_withl_in[:, start_idx:start_idx + 2],
                    dlatents_withl_in[:, :n_content],
                    act=act)
                x = apply_st(x, sh_matrix, up=False, fmaps=128, act=act)
                with tf.variable_scope('After_Shearing_noised'):
                    x = noised_conv_layer(x, layer_idx=4, fmaps=128, kernel=3)
            # Translation layers.
            with tf.variable_scope('Translation'):
                start_idx = start_idx + 2
                t_matrix = get_t_matrix(
                    dlatents_withl_in[:, start_idx:start_idx + 2],
                    dlatents_withl_in[:, :n_content],
                    act=act)
                x = apply_st(x, t_matrix, up=False, fmaps=128, act=act)
                with tf.variable_scope('After_Translation_noised'):
                    if resolution_log2 >= 6:
                        x = noised_conv_layer(x,
                                              layer_idx=5,
                                              up=True,
                                              fmaps=128,
                                              kernel=3)
                    else:
                        x = noised_conv_layer(x,
                                              layer_idx=5,
                                              fmaps=128,
                                              kernel=3)

    with tf.variable_scope('64x64' if resolution_log2 >= 6 else '32x32'):
        with tf.variable_scope('LocalHeat_C_latents'):
            with tf.variable_scope('ControlAttHeat'):
                heat_start_idx = label_size + D_global_size + C_global_size + \
                    sb_C_global_size + C_local_hfeat_size
                att_heat = get_att_heat(x, nheat=C_local_heat_size, act=act)
                # C_local_heat latent [-2, 2] --> [0, 1]
                heat_modifier = (2 + dlatents_withl_in[:, heat_start_idx:heat_start_idx + \
                                                 C_local_heat_size]) / 4.
                heat_modifier = get_conditional_modifier(
                    heat_modifier, dlatents_withl_in[:, :n_content], act=act)
                heat_modifier = tf.reshape(
                    heat_modifier,
                    [tf.shape(heat_modifier)[0], C_local_heat_size, 1, 1])
                att_heat = att_heat * heat_modifier
                x = tf.concat([x, att_heat], axis=1)
            with tf.variable_scope('After_LocalHeat_noised'):
                x = noised_conv_layer(x, layer_idx=6, fmaps=128, kernel=3)
    y = torgb(x, y, num_channels=num_channels)

    # # Tail layers.
    # for res in range(6, resolution_log2 + 1):
    # with tf.variable_scope('%dx%d' % (res * 2, res * 2)):
    # x = apply_bias_act(conv2d_layer(x,
    # fmaps=128,
    # kernel=1,
    # up=True,
    # resample_kernel=resample_kernel),
    # act=act)
    # y = torgb(x, y, num_channels=num_channels)
    images_out = y
    assert images_out.dtype == tf.as_dtype(dtype)
    return tf.identity(images_out, name='images_out')


def upsample(y, resample_kernel):
    with tf.variable_scope('Upsample'):
        return upsample_2d(y, k=resample_kernel)


def torgb(x, y, num_channels):
    with tf.variable_scope('ToRGB'):
        t = apply_bias_act(conv2d_layer(x, fmaps=num_channels, kernel=1))
        return t if y is None else y + t


# Return rotation matrix
def get_r_matrix(r_latents, cond_latent, act='lrelu'):
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
def get_s_matrix(s_latents, cond_latent, act='lrelu'):
    # s_latents: [-2., 2.] -> [1, 3]
    with tf.variable_scope('Condition0'):
        cond = apply_bias_act(dense_layer(cond_latent, fmaps=128), act=act)
    with tf.variable_scope('Condition1'):
        cond = apply_bias_act(dense_layer(cond, fmaps=1), act='sigmoid')
    scale = (s_latents + 2.) * cond + 1.
    tt_00 = scale
    tt_01 = tf.zeros_like(scale)
    tt_02 = tf.zeros_like(scale)
    tt_10 = tf.zeros_like(scale)
    tt_11 = scale
    tt_12 = tf.zeros_like(scale)
    theta = tf.concat([tt_00, tt_01, tt_02, tt_10, tt_11, tt_12], axis=1)
    return theta


# Return shear matrix
def get_sh_matrix(sh_latents, cond_latent, act='lrelu'):
    # sh_latents[:, 0]: [-2., 2.] -> [-1., 1.]
    # sh_latents[:, 1]: [-2., 2.] -> [-1., 1.]
    with tf.variable_scope('Condition0x'):
        cond_x = apply_bias_act(dense_layer(cond_latent, fmaps=128), act=act)
    with tf.variable_scope('Condition1x'):
        cond_x = apply_bias_act(dense_layer(cond_x, fmaps=1), act='sigmoid')
    with tf.variable_scope('Condition0y'):
        cond_y = apply_bias_act(dense_layer(cond_latent, fmaps=128), act=act)
    with tf.variable_scope('Condition1y'):
        cond_y = apply_bias_act(dense_layer(cond_y, fmaps=1), act='sigmoid')
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
def get_t_matrix(t_latents, cond_latent, act='lrelu'):
    # t_latents[:, 0]: [-2., 2.] -> [-0.5, 0.5]
    # t_latents[:, 1]: [-2., 2.] -> [-0.5, 0.5]
    with tf.variable_scope('Condition0x'):
        cond_x = apply_bias_act(dense_layer(cond_latent, fmaps=128), act=act)
    with tf.variable_scope('Condition1x'):
        cond_x = apply_bias_act(dense_layer(cond_x, fmaps=1), act='sigmoid')
    with tf.variable_scope('Condition0y'):
        cond_y = apply_bias_act(dense_layer(cond_latent, fmaps=128), act=act)
    with tf.variable_scope('Condition1y'):
        cond_y = apply_bias_act(dense_layer(cond_y, fmaps=1), act='sigmoid')
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


def get_conditional_modifier(modifier, cond_latent, act='lrelu'):
    with tf.variable_scope('Condition0'):
        cond = apply_bias_act(dense_layer(cond_latent, fmaps=128), act=act)
    with tf.variable_scope('Condition1'):
        cond = apply_bias_act(dense_layer(cond,
                                          fmaps=modifier.shape.as_list()[1]),
                              act='sigmoid')
    modifier = modifier * cond
    return modifier


# Apply spatial transform
def apply_st(x,
             st_matrix,
             up=True,
             fmaps=128,
             resample_kernel=[1, 3, 3, 1],
             act='lrelu'):
    with tf.variable_scope('Transform'):
        x = tf.transpose(x, [0, 2, 3, 1])  # NCHW -> NHWC
        x = transformer(x, st_matrix, out_dims=x.shape.as_list()[1:3])
        x = tf.transpose(x, [0, 3, 1, 2])  # NHWC -> NCHW
    with tf.variable_scope('ConvMayUp'):
        x = apply_bias_act(conv2d_layer(x,
                                        fmaps=fmaps,
                                        kernel=3,
                                        up=up,
                                        resample_kernel=resample_kernel),
                           act=act)
    with tf.variable_scope('Conv'):
        x = apply_bias_act(conv2d_layer(x, fmaps=fmaps, kernel=3), act=act)
    return x


def get_att_heat(x, nheat, act):
    with tf.variable_scope('Conv'):
        x = apply_bias_act(conv2d_layer(x, fmaps=128, kernel=3), act=act)
    with tf.variable_scope('ConvAtt'):
        x = apply_bias_act(conv2d_layer(x, fmaps=1, kernel=3), act='sigmoid')
    return x
