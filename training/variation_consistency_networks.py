#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: variation_consistency_networks.py
# --- Creation Date: 03-02-2020
# --- Last Modified: Thu 05 Mar 2020 05:00:28 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Variation-Consistency Networks.
"""

import numpy as np
import pdb
import collections
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib import EasyDict
from dnnlib.tflib.ops.upfirdn_2d import upsample_2d, downsample_2d
from dnnlib.tflib.ops.upfirdn_2d import upsample_conv_2d, conv_downsample_2d
from dnnlib.tflib.ops.fused_bias_act import fused_bias_act
from training.networks_stylegan2 import get_weight, dense_layer, conv2d_layer
from training.networks_stylegan2 import apply_bias_act, naive_upsample_2d
from training.networks_stylegan2 import naive_downsample_2d, modulated_conv2d_layer
from training.networks_stylegan2 import minibatch_stddev_layer
from training.spatial_biased_extended_networks import torgb, get_conditional_modifier
from training.spatial_biased_extended_networks import get_att_heat
from training.spatial_biased_modular_networks import split_module_names, build_D_layers
from training.spatial_biased_modular_networks import build_C_global_layers
from training.spatial_biased_modular_networks import build_local_heat_layers, build_local_hfeat_layers
from training.spatial_biased_modular_networks import build_noise_layer, build_conv_layer
from stn.stn import spatial_transformer_network as transformer


#----------------------------------------------------------------------------
# Variation Consistenecy main Generator
def G_main_vc(
        latents_in,  # First input: Latent vectors (Z) [minibatch, latent_size].
        labels_in,  # Second input: Conditioning labels [minibatch, label_size].
        is_training=False,  # Network is under training? Enables and disables specific features.
        is_validation=False,  # Network is under validation? Chooses which value to use for truncation_psi.
        return_dlatents=False,  # Return dlatents in addition to the images?
        is_template_graph=False,  # True = template graph constructed by the Network class, False = actual evaluation.
        components=dnnlib.EasyDict(
        ),  # Container for sub-networks. Retained between calls.
        mapping_func='G_mapping_spatial_biased_dsp',  # Build func name for the mapping network.
        synthesis_func='G_synthesis_vc_modular',  # Build func name for the synthesis network.
        **kwargs):  # Arguments for sub-networks (mapping and synthesis).
    # Validate arguments.
    assert not is_training or not is_validation

    # Setup components.
    if 'synthesis' not in components:
        components.synthesis = tflib.Network(
            'G_vc_synthesis', func_name=globals()[synthesis_func], **kwargs)
    if 'mapping' not in components:
        components.mapping = tflib.Network('G_vc_mapping',
                                           func_name=globals()[mapping_func],
                                           dlatent_broadcast=None,
                                           **kwargs)

    # Setup variables.
    lod_in = tf.get_variable('lod', initializer=np.float32(0), trainable=False)

    # Evaluate mapping network.
    dlatents = components.mapping.get_output_for(latents_in,
                                                 labels_in,
                                                 is_training=is_training,
                                                 **kwargs)
    dlatents = tf.cast(dlatents, tf.float32)

    # Evaluate synthesis network.
    deps = []
    if 'lod' in components.synthesis.vars:
        deps.append(tf.assign(components.synthesis.vars['lod'], lod_in))
    with tf.control_dependencies(deps):
        images_out, feat_map = components.synthesis.get_output_for(
            dlatents,
            is_training=is_training,
            force_clean_graph=is_template_graph,
            **kwargs)

    # Return requested outputs.
    images_out = tf.identity(images_out, name='images_out')
    if return_dlatents:
        return images_out, feat_map, dlatents
    return images_out, feat_map


def G_mapping_spatial_biased_dsp(
        latents_in,  # First input: Latent vectors (Z) [minibatch, latent_size].
        labels_in,  # Second input: Conditioning labels [minibatch, label_size].
        latent_size=7,  # Latent vector (Z) dimensionality.
        label_size=0,  # Label dimensionality, 0 if no labels.
        mapping_nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        dtype='float32',  # Data type to use for activations and outputs.
        **_kwargs):  # Ignore unrecognized keyword args.

    # Inputs.
    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    latents_in = tf.cast(latents_in, dtype)
    labels_in = tf.cast(labels_in, dtype)
    x = latents_in

    with tf.variable_scope('LabelConcat'):
        x = tf.concat([labels_in, x], axis=1)

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return tf.identity(x, name='dlatents_out')


def G_synthesis_vc_modular(
        dlatents_withl_in,  # Input: Disentangled latents (W) [minibatch, label_size+dlatent_size].
        dlatent_size=7,  # Disentangled latent (W) dimensionality. Including discrete info, rotation, scaling, xy shearing, and xy translation.
        label_size=0,  # Label dimensionality, 0 if no labels.
        module_list=None,  # A list containing module names, which represent semantic latents (exclude labels).
        num_channels=1,  # Number of output color channels.
        resolution=64,  # Output resolution.
        fmap_base=16 <<
        10,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_min=1,  # Minimum number of feature maps in any layer.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        dtype='float32',  # Data type to use for activations and outputs.
        resample_kernel=[
            1, 3, 3, 1
        ],  # Low-pass filter to apply when resampling activations. None = no filtering.
        fused_modconv=True,  # Implement modulated_conv2d_layer() as a single fused op?
        use_noise=False,  # If noise is used in this dataset.
        randomize_noise=True,  # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
        single_const=True,  # If only use a single constant feature at the begining.
        where_feat_map=15,  # For F_loss, which layer of feat map to use.
        **_kwargs):  # Ignore unrecognized keyword args.
    '''
    Modularized variation-consistent network.
    '''
    resolution_log2 = int(np.log2(resolution))  # == 6 for resolution 64
    assert resolution == 2**resolution_log2 and resolution >= 4

    def nf(stage):
        return np.clip(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_min,
                       fmap_max)

    num_layers = resolution_log2 * 2 - 2  # == 10 for resolution 64

    act = nonlinearity
    images_out = None

    # Note that module_list may include modules not containing latents,
    # e.g. Conv layers (size in this case means number of conv layers).
    key_ls, size_ls, count_dlatent_size, n_content = split_module_names(
        module_list)
    print('In key_ls:', key_ls)
    print('In size_ls:', size_ls)
    print('In count_dlatent_size:', count_dlatent_size)
    if label_size > 0:
        key_ls.insert(0, 'Label')
        size_ls.insert(0, label_size)
        n_content += label_size
    # module_dict = collections.OrderedDict(zip(key_ls, size_ls))

    # Primary inputs.
    assert dlatent_size == count_dlatent_size
    dlatents_withl_in.set_shape([None, label_size + count_dlatent_size])
    dlatents_withl_in = tf.cast(dlatents_withl_in, dtype)

    # Early layers consists of 4x4 constant layer.
    y = None
    if single_const:
        with tf.variable_scope('4x4'):
            with tf.variable_scope('Const'):
                x = tf.get_variable(
                    'const',
                    shape=[1, 8, 4, 4],
                    initializer=tf.initializers.random_normal())
                x = tf.tile(tf.cast(x, dtype),
                            [tf.shape(dlatents_withl_in)[0], 1, 1, 1])
    else:
        with tf.variable_scope('4x4'):
            with tf.variable_scope('Const'):
                x = tf.get_variable(
                    'const',
                    shape=[n_content, 8, 4, 4],
                    initializer=tf.initializers.random_normal())

    subkwargs = EasyDict()
    subkwargs.update(dlatents_withl_in=dlatents_withl_in,
                     n_content=n_content,
                     act=act,
                     dtype=dtype,
                     resample_kernel=resample_kernel,
                     fused_modconv=fused_modconv,
                     use_noise=use_noise,
                     randomize_noise=randomize_noise)

    # Build modules by module_dict.
    start_idx = 0
    # print('module_dict:', module_dict)
    # for scope_idx, k in enumerate(module_dict):
    for scope_idx, k in enumerate(key_ls):
        if scope_idx == where_feat_map:
            feat_map = x
        if (k.startswith('Label')) or (k.startswith('D_global')):
            # e.g. {'Label': 3}, {'D_global': 3}
            x = build_D_layers(x,
                               name=k,
                               n_latents=size_ls[scope_idx],
                               start_idx=start_idx,
                               scope_idx=scope_idx,
                               single_const=single_const,
                               fmaps=nf(scope_idx),
                               **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k.startswith('C_global'):
            # e.g. {'C_global': 2}
            x = build_C_global_layers(x,
                                      name=k,
                                      n_latents=size_ls[scope_idx],
                                      start_idx=start_idx,
                                      scope_idx=scope_idx,
                                      **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k.startswith('C_nocond_global'):
            # e.g. {'C_nocond_global': 2}
            x = build_C_global_nocond_layers(x,
                                      name=k,
                                      n_latents=size_ls[scope_idx],
                                      start_idx=start_idx,
                                      scope_idx=scope_idx,
                                      fmaps=nf(scope_idx),
                                      **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k.startswith('C_local_heat'):
            # e.g. {'C_local_heat': 4}
            x = build_local_heat_layers(x,
                                        name=k,
                                        n_latents=size_ls[scope_idx],
                                        start_idx=start_idx,
                                        scope_idx=scope_idx,
                                        fmaps=nf(scope_idx),
                                        **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k.startswith('C_local_hfeat'):
            # e.g. {'C_local_hfeat_size': 4}
            x = build_local_hfeat_layers(x,
                                         name=k,
                                         n_latents=size_ls[scope_idx],
                                         start_idx=start_idx,
                                         scope_idx=scope_idx,
                                         fmaps=nf(scope_idx),
                                         **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k.startswith('Noise'):
            # e.g. {'Noise': 1}
            x = build_noise_layer(x,
                                  name=k,
                                  n_layers=size_ls[scope_idx],
                                  scope_idx=scope_idx,
                                  fmaps=nf(scope_idx),
                                  **subkwargs)
        elif k.startswith('Conv'):
            # e.g. {'Conv-up': 2}, {'Conv-id': 1}
            x = build_conv_layer(x,
                                 name=k,
                                 n_layers=size_ls[scope_idx],
                                 scope_idx=scope_idx,
                                 fmaps=nf(scope_idx),
                                 **subkwargs)
        else:
            raise ValueError('Unsupported module type: ' + k)

    y = torgb(x, y, num_channels=num_channels)
    images_out = y
    assert images_out.dtype == tf.as_dtype(dtype)
    return tf.identity(images_out,
                       name='images_out'), tf.identity(feat_map,
                                                       name='feat_map')


#----------------------------------------------------------------------------
# Variation-Consistency Head network.


def vc_head(
        fake1,  # First input: generated image from z [minibatch, channel, height, width].
        fake2,  # Second input: hidden features from z + delta(z) [minibatch, channel, height, width].
        num_channels=3,  # Number of input color channels. Overridden based on dataset.
        resolution=1024,  # Input resolution. Overridden based on dataset.
        dlatent_size=10,
        D_global_size=0,
        fmap_base=16 <<
        10,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_min=1,  # Minimum number of feature maps in any layer.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
        nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, 0 = disable.
        mbstd_num_features=1,  # Number of features for the minibatch standard deviation layer.
        dtype='float32',  # Data type to use for activations and outputs.
        resample_kernel=[
            1, 3, 3, 1
        ],  # Low-pass filter to apply when resampling activations. None = no filtering.
        connect_mode='concat',  # How fake1 and fake2 connected.
        **_kwargs):  # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4

    def nf(stage):
        return np.clip(int(fmap_base / (2.0**(stage * fmap_decay))), fmap_min,
                       fmap_max)

    assert architecture in ['orig', 'skip', 'resnet']
    act = nonlinearity

    fake1.set_shape([None, num_channels, resolution, resolution])
    fake2.set_shape([None, num_channels, resolution, resolution])
    fake1 = tf.cast(fake1, dtype)
    fake2 = tf.cast(fake2, dtype)
    if connect_mode == 'diff':
        images_in = fake1 - fake2
    elif connect_mode == 'concat':
        images_in = tf.concat([fake1, fake2], axis=1)

    # Building blocks for main layers.
    def fromrgb(x, y, res):  # res = 2..resolution_log2
        with tf.variable_scope('FromRGB'):
            t = apply_bias_act(conv2d_layer(y, fmaps=nf(res - 1), kernel=1),
                               act=act)
            return t if x is None else x + t

    def block(x, res):  # res = 2..resolution_log2
        t = x
        with tf.variable_scope('Conv0'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res - 1), kernel=3),
                               act=act)
        with tf.variable_scope('Conv1_down'):
            x = apply_bias_act(conv2d_layer(x,
                                            fmaps=nf(res - 2),
                                            kernel=3,
                                            down=True,
                                            resample_kernel=resample_kernel),
                               act=act)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t,
                                 fmaps=nf(res - 2),
                                 kernel=1,
                                 down=True,
                                 resample_kernel=resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        return x

    def downsample(y):
        with tf.variable_scope('Downsample'):
            return downsample_2d(y, k=resample_kernel)

    # Main layers.
    x = None
    y = images_in
    for res in range(resolution_log2, 2, -1):
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if architecture == 'skip' or res == resolution_log2:
                x = fromrgb(x, y, res)
            x = block(x, res)
            if architecture == 'skip':
                y = downsample(y)

    # Final layers.
    with tf.variable_scope('4x4'):
        if architecture == 'skip':
            x = fromrgb(x, y, 2)
        if mbstd_group_size > 1:
            with tf.variable_scope('MinibatchStddev'):
                x = minibatch_stddev_layer(x, mbstd_group_size,
                                           mbstd_num_features)
        with tf.variable_scope('Conv'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(1), kernel=3), act=act)
        with tf.variable_scope('Dense0'):
            x = apply_bias_act(dense_layer(x, fmaps=nf(0)), act=act)

    # Output layer with label conditioning from "Which Training Methods for GANs do actually Converge?"
    with tf.variable_scope('Output'):
        with tf.variable_scope('Dense_VC'):
            x = apply_bias_act(
                dense_layer(x,
                            fmaps=(D_global_size +
                                   (dlatent_size - D_global_size))))

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    return x

#----------------------------------------------------------------------------
# StyleGAN2 discriminator simpler version.

def D_stylegan2_simple(
    images_in,                          # First input: Images [minibatch, channel, height, width].
    labels_in,                          # Second input: Labels [minibatch, label_size].
    num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
    resolution          = 1024,         # Input resolution. Overridden based on dataset.
    label_size          = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
    fmap_base           = 16 << 10,     # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_min            = 1,            # Minimum number of feature maps in any layer.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.
    architecture        = 'resnet',     # Architecture: 'orig', 'skip', 'resnet'.
    nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
    mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
    mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer.
    dtype               = 'float32',    # Data type to use for activations and outputs.
    resample_kernel     = [1,3,3,1],    # Low-pass filter to apply when resampling activations. None = no filtering.
    **_kwargs):                         # Ignore unrecognized keyword args.

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return np.clip(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_min, fmap_max)
    assert architecture in ['orig', 'skip', 'resnet']
    act = nonlinearity

    images_in.set_shape([None, num_channels, resolution, resolution])
    labels_in.set_shape([None, label_size])
    images_in = tf.cast(images_in, dtype)
    labels_in = tf.cast(labels_in, dtype)

    # Building blocks for main layers.
    def fromrgb(x, y, res): # res = 2..resolution_log2
        with tf.variable_scope('FromRGB'):
            t = apply_bias_act(conv2d_layer(y, fmaps=nf(res-1), kernel=1), act=act)
            return t if x is None else x + t
    def block(x, res): # res = 2..resolution_log2
        t = x
        # with tf.variable_scope('Conv0'):
            # x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-1), kernel=3), act=act)
        with tf.variable_scope('Conv1_down'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-2), kernel=3, down=True, resample_kernel=resample_kernel), act=act)
        if architecture == 'resnet':
            with tf.variable_scope('Skip'):
                t = conv2d_layer(t, fmaps=nf(res-2), kernel=1, down=True, resample_kernel=resample_kernel)
                x = (x + t) * (1 / np.sqrt(2))
        return x
    def downsample(y):
        with tf.variable_scope('Downsample'):
            return downsample_2d(y, k=resample_kernel)

    # Main layers.
    x = None
    y = images_in
    for res in range(resolution_log2, 2, -1):
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            if architecture == 'skip' or res == resolution_log2:
                x = fromrgb(x, y, res)
            x = block(x, res)
            if architecture == 'skip':
                y = downsample(y)

    # Final layers.
    with tf.variable_scope('4x4'):
        if architecture == 'skip':
            x = fromrgb(x, y, 2)
        if mbstd_group_size > 1:
            with tf.variable_scope('MinibatchStddev'):
                x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
        with tf.variable_scope('Conv'):
            x = apply_bias_act(conv2d_layer(x, fmaps=nf(1), kernel=3), act=act)
        with tf.variable_scope('Dense0'):
            x = apply_bias_act(dense_layer(x, fmaps=nf(0)), act=act)

    # Output layer with label conditioning from "Which Training Methods for GANs do actually Converge?"
    with tf.variable_scope('Output'):
        x = apply_bias_act(dense_layer(x, fmaps=max(labels_in.shape[1], 1)))
        if labels_in.shape[1] > 0:
            x = tf.reduce_sum(x * labels_in, axis=1, keepdims=True)
    scores_out = x

    # Output.
    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')
    return scores_out


def build_C_global_nocond_layers(x,
                                 name,
                                 n_latents,
                                 start_idx,
                                 scope_idx,
                                 dlatents_withl_in,
                                 act,
                                 fused_modconv,
                                 fmaps=128,
                                 **kwargs):
    '''
    Build continuous latent layers, e.g. C_global layers.
    '''
    with tf.variable_scope(name + '-' + str(scope_idx)):
        with tf.variable_scope('Conv0'):
            C_global_latents = apply_bias_act(dense_layer(
                dlatents_withl_in[:, start_idx:start_idx + n_latents], fmaps=128),
                                  act=act)
        # C_global_latents = dlatents_withl_in[:, start_idx:start_idx +
                                             # n_latents]
        with tf.variable_scope('Modulate'):
            x = apply_bias_act(modulated_conv2d_layer(x,
                                                      C_global_latents,
                                                      fmaps=fmaps,
                                                      kernel=3,
                                                      up=False,
                                                      fused_modconv=fused_modconv),
                               act=act)
    return x
