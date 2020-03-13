#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: spatial_biased_modular_networks.py
# --- Creation Date: 01-02-2020
# --- Last Modified: Sat 15 Feb 2020 02:09:54 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Spatial-Biased Network with modular implementation.
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

LATENT_MODULES = [
    'D_global', 'C_nocond_global', 'C_global', 'SB', 'C_local_heat', 'C_local_hfeat'
]


def G_synthesis_sb_modular(
        dlatents_withl_in,  # Input: Disentangled latents (W) [minibatch, label_size+dlatent_size].
        dlatent_size=7,  # Disentangled latent (W) dimensionality. Including discrete info, rotation, scaling, xy shearing, and xy translation.
        label_size=0,  # Label dimensionality, 0 if no labels.
        module_list=None,  # A list containing module names, which represent semantic latents (exclude labels).
        num_channels=1,  # Number of output color channels.
        resolution=64,  # Output resolution.
        nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu', etc.
        dtype='float32',  # Data type to use for activations and outputs.
        resample_kernel=[
            1, 3, 3, 1
        ],  # Low-pass filter to apply when resampling activations. None = no filtering.
        fused_modconv=True,  # Implement modulated_conv2d_layer() as a single fused op?
        use_noise=False,  # If noise is used in this dataset.
        randomize_noise=True,  # True = randomize noise inputs every time (non-deterministic), False = read noise inputs from variables.
        single_const=True,
        **_kwargs):  # Ignore unrecognized keyword args.
    '''
    Modularized spatial-biased network.
    '''
    resolution_log2 = int(np.log2(resolution))  # == 6 for resolution 64
    assert resolution == 2**resolution_log2 and resolution >= 4
    num_layers = resolution_log2 * 2 - 2  # == 10 for resolution 64

    act = nonlinearity
    images_out = None

    # Note that module_list may include modules not containing latents,
    # e.g. Conv layers (size in this case means number of conv layers).
    key_ls, size_ls, count_dlatent_size, n_content = split_module_names(
        module_list)
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
                    shape=[1, 128, 4, 4],
                    initializer=tf.initializers.random_normal())
                x = tf.tile(tf.cast(x, dtype),
                            [tf.shape(dlatents_withl_in)[0], 1, 1, 1])
    else:
        with tf.variable_scope('4x4'):
            with tf.variable_scope('Const'):
                x = tf.get_variable(
                    'const',
                    shape=[n_content, 128, 4, 4],
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
        if (k.startswith('Label')) or (k.startswith('D_global')):
            # e.g. {'Label': 3}, {'D_global': 3}
            x = build_D_layers(x,
                               name=k,
                               n_latents=size_ls[scope_idx],
                               start_idx=start_idx,
                               scope_idx=scope_idx,
                               single_const=single_const,
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
        elif k.startswith('SB'):
            # e.g. {'SB-rotation: 1}, {'SB-shearing': 2}
            x = build_SB_layers(x,
                                name=k,
                                n_latents=size_ls[scope_idx],
                                start_idx=start_idx,
                                scope_idx=scope_idx,
                                **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k.startswith('C_local_heat'):
            # e.g. {'C_local_heat': 4}
            x = build_local_heat_layers(x,
                                        name=k,
                                        n_latents=size_ls[scope_idx],
                                        start_idx=start_idx,
                                        scope_idx=scope_idx,
                                        **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k.startswith('C_local_hfeat'):
            # e.g. {'C_local_hfeat_size': 4}
            x = build_local_hfeat_layers(x,
                                         name=k,
                                         n_latents=size_ls[scope_idx],
                                         start_idx=start_idx,
                                         scope_idx=scope_idx,
                                         **subkwargs)
            start_idx += size_ls[scope_idx]
        elif k.startswith('Noise'):
            # e.g. {'Noise': 1}
            x = build_noise_layer(x,
                                  name=k,
                                  n_layers=size_ls[scope_idx],
                                  scope_idx=scope_idx,
                                  **subkwargs)
        elif k.startswith('Conv'):
            # e.g. {'Conv-up': 2}, {'Conv-id': 1}
            x = build_conv_layer(x,
                                 name=k,
                                 n_layers=size_ls[scope_idx],
                                 scope_idx=scope_idx,
                                 **subkwargs)
        else:
            raise ValueError('Unsupported module type: ' + k)

    y = torgb(x, y, num_channels=num_channels)
    images_out = y
    assert images_out.dtype == tf.as_dtype(dtype)
    return tf.identity(images_out, name='images_out')


def split_module_names(module_list, **kwargs):
    '''
    Split the input module_list.
    e.g. ['D_global-0', 'C_global-2', 'Conv-id-1',
            'Conv-up-1', 'SB-rotation-0', 'SB_scaling-2',
            'C_local_heat-2', 'Noise-1', 'C_local_hfeat-2',
            'SB_magnification-1', 'Conv-up-1', 'Noise-1', 'Conv-id-1',
            'SB-shearing-2', 'SB-translation-2', 'Conv-up-1',
            'C_local_heat-2', 'Conv-up-1', 'C_local_hfeat-1',
            'Noise-1', 'Conv-id-1']
    We assume all content_latents are in the begining.
    '''
    key_ls = []
    size_ls = []
    count_dlatent_size = 0
    n_content = 0
    # print('In split:', module_list)
    for module in module_list:
        m_name = module.split('-')[0]
        m_key = '-'.join(module.split('-')[:-1])  # exclude size
        size = int(module.split('-')[-1])
        if size > 0:
            if m_name in LATENT_MODULES:
                count_dlatent_size += size
            key_ls.append(m_key)
            size_ls.append(size)
        # if m_name in ['D_global', 'C_global']:
        if m_name in ['D_global']:
            n_content += size
    return key_ls, size_ls, count_dlatent_size, n_content


def build_D_layers(x,
                   name,
                   n_latents,
                   start_idx,
                   scope_idx,
                   single_const,
                   dlatents_withl_in,
                   act,
                   fused_modconv,
                   fmaps=128,
                   **kwargs):
    '''
    Build discrete latent layers including label and D_global layers.
    '''
    with tf.variable_scope(name + '-' + str(scope_idx)):
        if single_const:
            x = apply_bias_act(modulated_conv2d_layer(
                x,
                dlatents_withl_in[:, start_idx:start_idx + n_latents],
                fmaps=fmaps,
                kernel=3,
                up=False,
                fused_modconv=fused_modconv),
                               act=act)
        else:
            # soft version
            # x_softmax = tf.nn.softmax(
            # dlatents_withl_in[:, start_idx:start_idx + n_latents], axis=1)
            # x_softmax = tf.reshape(
            # x_softmax, [tf.shape(x_softmax)[0], n_latents, 1, 1, 1])
            # x = tf.reduce_sum(x * x_softmax, axis=1)
            # print('x.shape:', x.shape.as_list())

            # hard version
            x_indices = tf.argmax(dlatents_withl_in[:, start_idx:start_idx +
                                                    n_latents],
                                  axis=1)
            print('x_indices.shape:', x_indices.shape.as_list())
            print('before gather_nd x.shape:', x.shape.as_list())
            x = tf.gather(x, x_indices, axis=0)
            print('after gather_nd x.shape:', x.shape.as_list())
    return x


def build_C_global_layers(x,
                          name,
                          n_latents,
                          start_idx,
                          scope_idx,
                          dlatents_withl_in,
                          n_content,
                          act,
                          fused_modconv,
                          fmaps=128,
                          **kwargs):
    '''
    Build continuous latent layers, e.g. C_global layers.
    '''
    with tf.variable_scope(name + '-' + str(scope_idx)):
        if n_content > 0:
            with tf.variable_scope('Condition0'):
                cond = apply_bias_act(dense_layer(
                    dlatents_withl_in[:, :n_content], fmaps=128),
                                      act=act)
            with tf.variable_scope('Condition1'):
                cond = apply_bias_act(dense_layer(cond, fmaps=n_latents),
                                      act='sigmoid')
        else:
            cond = 1.
        C_global_latents = dlatents_withl_in[:, start_idx:start_idx +
                                             n_latents] * cond
        x = apply_bias_act(modulated_conv2d_layer(x,
                                                  C_global_latents,
                                                  fmaps=fmaps,
                                                  kernel=3,
                                                  up=False,
                                                  fused_modconv=fused_modconv),
                           act=act)
    return x


def build_SB_layers(x,
                    name,
                    n_latents,
                    start_idx,
                    scope_idx,
                    dlatents_withl_in,
                    n_content,
                    act,
                    resample_kernel,
                    fused_modconv,
                    fmaps=128,
                    **kwargs):
    '''
    Build spatial-biased networks modules.
    e.g. ['SB-rotation-0', 'SB_scaling-1', 'SB_magnification-1',
            'SB-shearing-2', 'SB-translation-2']
    '''
    sb_type = name.split('-')[-1]
    assert sb_type in [
        'rotation', 'scaling', 'magnification', 'shearing', 'translation'
    ]
    with tf.variable_scope(name + '-' + str(scope_idx)):
        if sb_type == 'magnification':
            assert n_latents == 1
            # [-2., 2.] --> [0.5, 1.]
            magnifier = (dlatents_withl_in[:, start_idx:start_idx + n_latents]
                         + 6.) / 8.
            magnifier = tf.reshape(magnifier,
                                   [tf.shape(magnifier)[0], 1, 1, 1])
            x *= magnifier
        else:
            if sb_type == 'rotation':
                assert n_latents == 1
                theta = get_r_matrix(dlatents_withl_in[:, start_idx:start_idx +
                                                       n_latents],
                                     dlatents_withl_in[:, :n_content],
                                     act=act)
            elif sb_type == 'scaling':
                assert n_latents <= 2
                theta = get_s_matrix(dlatents_withl_in[:, start_idx:start_idx +
                                                       n_latents],
                                     dlatents_withl_in[:, :n_content],
                                     act=act)
            elif sb_type == 'shearing':
                assert n_latents <= 2
                theta = get_sh_matrix(
                    dlatents_withl_in[:, start_idx:start_idx + n_latents],
                    dlatents_withl_in[:, :n_content],
                    act=act)
            elif sb_type == 'translation':
                assert n_latents <= 2
                theta = get_t_matrix(dlatents_withl_in[:, start_idx:start_idx +
                                                       n_latents],
                                     dlatents_withl_in[:, :n_content],
                                     act=act)
            x = apply_st(x, theta)
    return x


def build_local_heat_layers(x, name, n_latents, start_idx, scope_idx,
                            dlatents_withl_in, n_content, act, **kwargs):
    '''
    Build local heatmap layers. They control local strength by attention maps.
    '''
    with tf.variable_scope(name + '-' + str(scope_idx)):
        att_heat = get_att_heat(x, nheat=n_latents, act=act)
        # C_local_heat latent [-2, 2] --> [0, 1]
        heat_modifier = (
            2 + dlatents_withl_in[:, start_idx:start_idx + n_latents]) / 4.
        heat_modifier = get_conditional_modifier(
            heat_modifier, dlatents_withl_in[:, :n_content], act=act)
        heat_modifier = tf.reshape(
            heat_modifier, [tf.shape(heat_modifier)[0], n_latents, 1, 1])
        att_heat = att_heat * heat_modifier
        x = tf.concat([x, att_heat], axis=1)
    return x


def build_local_hfeat_layers(x, name, n_latents, start_idx, scope_idx,
                             dlatents_withl_in, n_content, act, dtype,
                             **kwargs):
    '''
    Build local heatmap*features.
    They contorl local presence of a feature by attention maps.
    '''
    with tf.variable_scope(name + '-' + str(scope_idx)):
        with tf.variable_scope('ConstFeats'):
            const_feats = tf.get_variable(
                'constfeats',
                shape=[1, n_latents, 32, 1, 1],
                initializer=tf.initializers.random_normal())
            const_feats = tf.tile(tf.cast(const_feats, dtype),
                                  [tf.shape(const_feats)[0], 1, 1, 1, 1])
        with tf.variable_scope('ControlAttHeat'):
            att_heat = get_att_heat(x, nheat=n_latents, act=act)
            att_heat = tf.reshape(att_heat,
                                  [tf.shape(att_heat)[0], n_latents, 1] +
                                  att_heat.shape.as_list()[2:4])
            # C_local_heat latent [-2, 2] --> [0, 1]
            hfeat_modifier = (
                2 + dlatents_withl_in[:, start_idx:start_idx + n_latents]) / 4.
            hfeat_modifier = get_conditional_modifier(
                hfeat_modifier, dlatents_withl_in[:, :n_content], act=act)
            hfeat_modifier = tf.reshape(hfeat_modifier,
                                        [tf.shape(x)[0], n_latents, 1, 1, 1])
            att_heat = att_heat * hfeat_modifier
            added_feats = const_feats * att_heat
            added_feats = tf.reshape(added_feats, [
                tf.shape(att_heat)[0], n_latents * att_heat.shape.as_list()[2]
            ] + att_heat.shape.as_list()[3:5])
            x = tf.concat([x, added_feats], axis=1)
    return x


def build_noise_layer(x,
                      name,
                      n_layers,
                      scope_idx,
                      act,
                      use_noise,
                      randomize_noise,
                      fmaps=128,
                      **kwargs):
    for i in range(n_layers):
        with tf.variable_scope(name + '-' + str(scope_idx) + '-' + str(i)):
            x = conv2d_layer(x, fmaps=fmaps, kernel=3, up=False)
            if use_noise:
                if randomize_noise:
                    noise = tf.random_normal(
                        [tf.shape(x)[0], 1, x.shape[2], x.shape[3]],
                        dtype=x.dtype)
                else:
                    # noise = tf.get_variable(
                    # 'noise_variable-' + str(scope_idx) + '-' + str(i),
                    # shape=[1, 1, x.shape[2], x.shape[3]],
                    # initializer=tf.initializers.random_normal(),
                    # trainable=False)
                    noise_np = np.random.normal(size=(1, 1, x.shape[2],
                                                      x.shape[3]))
                    noise = tf.constant(noise_np)
                    noise = tf.cast(noise, x.dtype)
                noise_strength = tf.get_variable(
                    'noise_strength-' + str(scope_idx) + '-' + str(i),
                    shape=[],
                    initializer=tf.initializers.zeros())
                x += noise * tf.cast(noise_strength, x.dtype)
            x = apply_bias_act(x, act=act)
    return x


def build_conv_layer(x,
                     name,
                     n_layers,
                     scope_idx,
                     act,
                     resample_kernel,
                     fmaps=128,
                     **kwargs):
    # e.g. {'Conv-up': 2}, {'Conv-id': 1}
    sample_type = name.split('-')[-1]
    assert sample_type in ['up', 'down', 'id']
    for i in range(n_layers):
        with tf.variable_scope(name + '-' + str(scope_idx) + '-' + str(i)):
            x = apply_bias_act(conv2d_layer(x,
                                            fmaps=fmaps,
                                            kernel=3,
                                            up=(sample_type == 'up'),
                                            down=(sample_type == 'down'),
                                            resample_kernel=resample_kernel),
                               act=act)
    return x


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
    # s_latents[:, 0]: [-2., 2.] -> [1., 3.]
    # s_latents[:, 1]: [-2., 2.] -> [1., 3.]
    if s_latents.shape.as_list()[1] == 1:
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
    else:
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
        scale = (s_latents + 2.) * cond + 1.
        tt_00 = scale[:, 0:1]
        tt_01 = tf.zeros_like(scale[:, 0:1])
        tt_02 = tf.zeros_like(scale[:, 0:1])
        tt_10 = tf.zeros_like(scale[:, 1:])
        tt_11 = scale[:, 1:]
        tt_12 = tf.zeros_like(scale[:, 1:])
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
    if t_latents.shape.as_list()[1] == 1:
        with tf.variable_scope('Condition0x'):
            cond = apply_bias_act(dense_layer(cond_latent, fmaps=128), act=act)
        with tf.variable_scope('Condition1x'):
            cond = apply_bias_act(dense_layer(cond, fmaps=1), act='sigmoid')
        xy_shift = t_latents / 4. * cond
        tt_00 = tf.ones_like(xy_shift)
        tt_01 = tf.zeros_like(xy_shift)
        tt_02 = xy_shift
        tt_10 = tf.zeros_like(xy_shift)
        tt_11 = tf.ones_like(xy_shift)
        tt_12 = xy_shift
    else:
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


def apply_st(x, st_matrix):
    with tf.variable_scope('Transform'):
        x = tf.transpose(x, [0, 2, 3, 1])  # NCHW -> NHWC
        x = transformer(x, st_matrix, out_dims=x.shape.as_list()[1:3])
        x = tf.transpose(x, [0, 3, 1, 2])  # NHWC -> NCHW
    return x
