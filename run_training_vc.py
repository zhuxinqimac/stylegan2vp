#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: run_training_vc.py
# --- Creation Date: 04-02-2020
# --- Last Modified: Fri 13 Mar 2020 17:37:55 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Run training file for variation consistency related networks use.
Code borrowed from run_training.py from NVIDIA.
"""

import argparse
import copy
import os
import sys

import dnnlib
from dnnlib import EasyDict

from metrics.metric_defaults import metric_defaults
from training.variation_consistency_networks import split_module_names, LATENT_MODULES

#----------------------------------------------------------------------------

_valid_configs = [
    # Table 1
    'config-a',  # Baseline StyleGAN
    'config-b',  # + Weight demodulation
    'config-c',  # + Lazy regularization
    'config-d',  # + Path length regularization
    'config-e',  # + No growing, new G & D arch.
    'config-f',  # + Large networks (default)

    # Table 2
    'config-e-Gorig-Dorig',
    'config-e-Gorig-Dresnet',
    'config-e-Gorig-Dskip',
    'config-e-Gresnet-Dorig',
    'config-e-Gresnet-Dresnet',
    'config-e-Gresnet-Dskip',
    'config-e-Gskip-Dorig',
    'config-e-Gskip-Dresnet',
    'config-e-Gskip-Dskip',
]

#----------------------------------------------------------------------------


def run(dataset, data_dir, result_dir, config_id, num_gpus, total_kimg, gamma,
        mirror_augment, metrics, resume_pkl, 
        fmap_decay=0.15, D_lambda=1, C_lambda=1, F_beta=0, cls_alpha=0, 
        n_samples_per=10, module_list=None, single_const=True, model_type='spatial_biased', 
        epsilon_loss=0.4, where_feat_map=15, random_eps=False, latent_type='uniform', 
        delta_type='onedim', cascading=False, connect_mode='concat'):
    # print('module_list:', module_list)
    train = EasyDict(run_func_name='training.training_loop_vc.training_loop_vc'
                     )  # Options for training loop.

    D_global_size = 0
    if model_type == 'info_gan':
        module_list = _str_to_list(module_list)
        key_ls, size_ls, count_dlatent_size, _ = split_module_names(
            module_list)
        for i, key in enumerate(key_ls):
            if key.startswith('D_global') or key.startswith('D_nocond_global'):
                D_global_size += size_ls[i]
                break
        print('D_global_size:', D_global_size)
        print('key_ls:', key_ls)
        print('size_ls:', size_ls)
        print('count_dlatent_size:', count_dlatent_size)
        G = EasyDict(
            func_name=
            'training.variation_consistency_networks.G_main_vc',
            synthesis_func='G_synthesis_vc_modular',
            fmap_min=16, fmap_max=512, fmap_decay=fmap_decay, latent_size=count_dlatent_size, 
            dlatent_size=count_dlatent_size, D_global_size=D_global_size, 
            module_list=module_list, single_const=single_const, 
            where_feat_map=where_feat_map, use_noise=True)  # Options for generator network.
        # I = EasyDict(func_name='training.info_gan_networks.info_gan_head',
                     # dlatent_size=count_dlatent_size, D_global_size=D_global_size,
                     # fmap_decay=fmap_decay, fmap_min=16, fmap_max=512)
        I = EasyDict(func_name='training.info_gan_networks.info_gan_body',
                     dlatent_size=count_dlatent_size, 
                     D_global_size=D_global_size, fmap_max=512)
        D = EasyDict(
            func_name='training.info_gan_networks.D_info_gan_stylegan2',
            fmap_max=512)  # Options for discriminator network.
        I_info = EasyDict()
        desc = 'info_gan_net'
    elif model_type == 'vc_gan_with_vc_head':
        module_list = _str_to_list(module_list)
        key_ls, size_ls, count_dlatent_size, _ = split_module_names(
            module_list)
        for i, key in enumerate(key_ls):
            if key.startswith('D_global') or key.startswith('D_nocond_global'):
                D_global_size += size_ls[i]
                break
        print('D_global_size:', D_global_size)
        print('key_ls:', key_ls)
        print('size_ls:', size_ls)
        print('count_dlatent_size:', count_dlatent_size)
        G = EasyDict(
            func_name='training.variation_consistency_networks.G_main_vc',
            synthesis_func='G_synthesis_vc_modular',
            fmap_min=16, fmap_max=512, fmap_decay=fmap_decay, latent_size=count_dlatent_size,
            dlatent_size=count_dlatent_size, D_global_size=D_global_size,
            module_list=module_list, single_const=single_const,
            where_feat_map=where_feat_map, use_noise=True)  # Options for generator network.
        I = EasyDict(func_name='training.variation_consistency_networks.vc_head',
            dlatent_size=count_dlatent_size, D_global_size=D_global_size, fmap_max=512, 
            connect_mode=connect_mode)
        D = EasyDict(func_name='training.networks_stylegan2.D_stylegan2',
            fmap_max=512)  # Options for discriminator network.
        # D = EasyDict(func_name='training.variation_consistency_networks.D_stylegan2_simple',
                     # fmap_max=512)  # Options for discriminator network.
        I_info = EasyDict()
        desc = 'vc_gan_with_vc_head_net'
    elif model_type == 'vc_gan_with_vc_head_with_cls':
        module_list = _str_to_list(module_list)
        key_ls, size_ls, count_dlatent_size, _ = split_module_names(
            module_list)
        for i, key in enumerate(key_ls):
            if key.startswith('D_global') or key.startswith('D_nocond_global'):
                D_global_size += size_ls[i]
                break
        print('D_global_size:', D_global_size)
        print('key_ls:', key_ls)
        print('size_ls:', size_ls)
        print('count_dlatent_size:', count_dlatent_size)
        G = EasyDict(
            func_name='training.variation_consistency_networks.G_main_vc',
            synthesis_func='G_synthesis_vc_modular',
            fmap_min=16, fmap_max=512, fmap_decay=fmap_decay, latent_size=count_dlatent_size,
            dlatent_size=count_dlatent_size, D_global_size=D_global_size,
            module_list=module_list, single_const=single_const,
            where_feat_map=where_feat_map, use_noise=True)  # Options for generator network.
        I = EasyDict(func_name='training.variation_consistency_networks.vc_head',
            dlatent_size=count_dlatent_size, D_global_size=D_global_size, fmap_max=512, 
            connect_mode=connect_mode)
        I_info = EasyDict(func_name='training.info_gan_networks.info_gan_head_cls',
                     dlatent_size=count_dlatent_size, D_global_size=D_global_size,
                     fmap_decay=fmap_decay, fmap_min=16, fmap_max=512)
        D = EasyDict(
            func_name='training.info_gan_networks.D_info_gan_stylegan2',
            fmap_max=512)  # Options for discriminator network.
        desc = 'vc_gan_with_vc_head_net_with_cls'
    elif model_type == 'vc_gan':
        module_list = _str_to_list(module_list)
        key_ls, size_ls, count_dlatent_size, _ = split_module_names(
            module_list)
        for i, key in enumerate(key_ls):
            if key.startswith('D_global') or key.startswith('D_nocond_global'):
                D_global_size += size_ls[i]
                break
        print('D_global_size:', D_global_size)
        print('key_ls:', key_ls)
        print('size_ls:', size_ls)
        print('count_dlatent_size:', count_dlatent_size)
        G = EasyDict(
            func_name='training.variation_consistency_networks.G_main_vc',
            # func_name='training.spatial_biased_networks.G_main_spatial_biased_dsp',
            synthesis_func='G_synthesis_vc_modular',
            fmap_min=16, fmap_max=512, fmap_decay=fmap_decay, latent_size=count_dlatent_size,
            dlatent_size=count_dlatent_size, D_global_size=D_global_size, 
            module_list=module_list, single_const=single_const, 
            where_feat_map=where_feat_map, use_noise=True)  # Options for generator network.
        I = EasyDict()
        I_info = EasyDict()
        D = EasyDict(func_name='training.networks_stylegan2.D_stylegan2',
                     fmap_max=512)  # Options for discriminator network.
        # D = EasyDict(func_name='training.variation_consistency_networks.D_stylegan2_simple',
                     # fmap_max=512)  # Options for discriminator network.
        # D         = EasyDict(func_name='training.spatial_biased_networks.D_with_discrete_dsp', fmap_max=128)  # Options for discriminator network.
        desc = 'vc_gan_net'
    else:
        raise ValueError('Not supported model tyle: ' + model_type)

    G_opt = EasyDict(beta1=0.0, beta2=0.99,
                     epsilon=1e-8)  # Options for generator optimizer.
    D_opt = EasyDict(beta1=0.0, beta2=0.99,
                     epsilon=1e-8)  # Options for discriminator optimizer.
    if model_type == 'info_gan':
        G_loss = EasyDict(func_name='training.loss.G_logistic_ns_info_gan',
            D_global_size=D_global_size, D_lambda=D_lambda, C_lambda=C_lambda,
            latent_type=latent_type)  # Options for generator loss.
        D_loss = EasyDict(func_name='training.loss.D_logistic_r1_info_gan',
            D_global_size=D_global_size, latent_type=latent_type)  # Options for discriminator loss.
    elif model_type == 'vc_gan_with_vc_head':
        G_loss = EasyDict(func_name='training.loss.G_logistic_ns_vc',
            D_global_size=D_global_size, C_lambda=C_lambda, F_beta=F_beta,
            epsilon=epsilon_loss, random_eps=random_eps, latent_type=latent_type, 
            delta_type=delta_type, cascading=cascading)  # Options for generator loss.
        D_loss = EasyDict(func_name='training.loss.D_logistic_r1_dsp',
            D_global_size=D_global_size, latent_type=latent_type)  # Options for discriminator loss.
    elif model_type == 'vc_gan_with_vc_head_with_cls':
        G_loss = EasyDict(func_name='training.loss.G_logistic_ns_vc',
            D_global_size=D_global_size, C_lambda=C_lambda, F_beta=F_beta, cls_alpha=cls_alpha,
            epsilon=epsilon_loss, random_eps=random_eps, latent_type=latent_type, 
            delta_type=delta_type, cascading=cascading)  # Options for generator loss.
        D_loss = EasyDict(func_name='training.loss.D_logistic_r1_info_gan',
            D_global_size=D_global_size, latent_type=latent_type)  # Options for discriminator loss.
    else:
        G_loss = EasyDict(func_name='training.loss.G_logistic_ns_dsp',
            D_global_size=D_global_size, latent_type=latent_type)  # Options for generator loss.
        D_loss = EasyDict(func_name='training.loss.D_logistic_r1_dsp',
            D_global_size=D_global_size, latent_type=latent_type)  # Options for discriminator loss.
    sched = EasyDict()  # Options for TrainingSchedule.
    grid = EasyDict(size='1080p', layout='random')  # Options for setup_snapshot_image_grid().
    sc = dnnlib.SubmitConfig()  # Options for dnnlib.submit_run().
    tf_config = {'rnd.np_random_seed': 1000}  # Options for tflib.init_tf().

    train.data_dir = data_dir
    train.total_kimg = total_kimg
    train.mirror_augment = mirror_augment
    train.image_snapshot_ticks = train.network_snapshot_ticks = 10
    sched.G_lrate_base = sched.D_lrate_base = 0.002
    sched.minibatch_size_base = 32
    sched.minibatch_gpu_base = 16
    D_loss.gamma = 10
    metrics = [metric_defaults[x] for x in metrics]

    desc += '-' + dataset
    dataset_args = EasyDict(tfrecord_dir=dataset, max_label_size='full')

    assert num_gpus in [1, 2, 4, 8]
    sc.num_gpus = num_gpus
    desc += '-%dgpu' % num_gpus

    assert config_id in _valid_configs
    desc += '-' + config_id

    # Configs A-E: Shrink networks to match original StyleGAN.
    # if config_id != 'config-f':
    # G.fmap_base = D.fmap_base = 8 << 10
    I.fmap_base = G.fmap_base = D.fmap_base = 2 << 8

    # Config E: Set gamma to 100 and override G & D architecture.
    if config_id.startswith('config-e'):
        D_loss.gamma = 100
        if 'Gorig' in config_id: G.architecture = 'orig'
        if 'Gskip' in config_id: G.architecture = 'skip'  # (default)
        if 'Gresnet' in config_id: G.architecture = 'resnet'
        if 'Dorig' in config_id: D.architecture = 'orig'
        if 'Dskip' in config_id: D.architecture = 'skip'
        if 'Dresnet' in config_id: D.architecture = 'resnet'  # (default)

    if gamma is not None:
        D_loss.gamma = gamma

    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    kwargs = EasyDict(train)
    kwargs.update(G_args=G, D_args=D, I_args=I, I_info_args=I_info, G_opt_args=G_opt, D_opt_args=D_opt,
                  G_loss_args=G_loss, D_loss_args=D_loss,
                  use_info_gan=(model_type == 'info_gan'),
                  use_vc_head=(model_type == 'vc_gan_with_vc_head'),
                  use_vc_head_with_cls=(model_type == 'vc_gan_with_vc_head_with_cls'),
                  traversal_grid=True)
    n_continuous = 0
    for i, key in enumerate(key_ls):
        m_name = key.split('-')[0]
        if (m_name in LATENT_MODULES) and (not m_name == 'D_global'):
            n_continuous += size_ls[i]

    kwargs.update(dataset_args=dataset_args, sched_args=sched, grid_args=grid, metric_arg_list=metrics,
                  tf_config=tf_config, resume_pkl=resume_pkl, n_discrete=D_global_size,
                  n_continuous=n_continuous, n_samples_per=n_samples_per)
    kwargs.submit_config = copy.deepcopy(sc)
    kwargs.submit_config.run_dir_root = result_dir
    kwargs.submit_config.run_desc = desc
    dnnlib.submit_run(**kwargs)


#----------------------------------------------------------------------------


def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _str_to_list(v):
    v_values = v.strip()[1:-1]
    module_list = [x.strip() for x in v_values.split(',')]
    return module_list


def _parse_comma_sep(s):
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')


#----------------------------------------------------------------------------

_examples = '''examples:

  # Train spatial-biased net using the dsprites dataset
  CUDA_VISIBLE_DEVICES=1 python %(prog)s --num-gpus=1 \
  --data-dir=/mnt/hdd/Datasets/dsprites --dataset=dsprites_nolabel_tfr

valid configs:

  ''' + ', '.join(_valid_configs) + '''

valid metrics:

  ''' + ', '.join(sorted([x for x in metric_defaults.keys()])) + '''

'''


def main():
    parser = argparse.ArgumentParser(
        description='Train VCGAN and INFOGAN.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--result-dir',
        help='Root directory for run results (default: %(default)s)',
        default='results',
        metavar='DIR')
    parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    parser.add_argument('--dataset', help='Training dataset', required=True)
    parser.add_argument('--config', help='Training config (default: %(default)s)',
                        default='config-e', dest='config_id', metavar='CONFIG')
    parser.add_argument('--num-gpus', help='Number of GPUs (default: %(default)s)',
                        default=1, type=int, metavar='N')
    parser.add_argument('--total-kimg',
        help='Training length in thousands of images (default: %(default)s)',
        metavar='KIMG', default=25000, type=int)
    parser.add_argument('--gamma',
        help='R1 regularization weight (default is config dependent)',
        default=None, type=float)
    parser.add_argument('--mirror-augment', help='Mirror augment (default: %(default)s)',
                        default=False, metavar='BOOL', type=_str_to_bool)
    parser.add_argument(
        '--metrics', help='Comma-separated list of metrics or "none" (default: %(default)s)',
        default='None', type=_parse_comma_sep)
    parser.add_argument('--model_type', help='Type of model to train', default='vc_gan',
                        type=str, metavar='MODEL_TYPE', choices=['info_gan', 'vc_gan', 'vc_gan_with_vc_head', 'vc_gan_with_vc_head_with_cls'])
    parser.add_argument('--resume_pkl', help='Continue training using pretrained pkl.',
                        default=None, metavar='RESUME_PKL', type=str)
    parser.add_argument('--n_samples_per', help='Number of samples for each line in traversal (default: %(default)s)',
        metavar='N_SHOWN_SAMPLES_PER_LINE', default=10, type=int)
    parser.add_argument('--module_list', help='Module list for modular network.',
                        default=None, metavar='MODULE_LIST', type=str)
    parser.add_argument(
        '--single_const',
        help='Use a single constant feature at the top or not (if not, n_classes of const feature maps will be used and gathered).',
        default=True, metavar='BOOL', type=_str_to_bool)
    parser.add_argument('--D_lambda', help='Discrete lambda for INFO-GAN and VC-GAN.',
                        metavar='D_LAMBDA', default=1, type=float)
    parser.add_argument('--C_lambda', help='Continuous lambda for INFO-GAN and VC-GAN.',
                        metavar='C_LAMBDA', default=1, type=float)
    parser.add_argument('--F_beta', help='F_beta INFO-GAN and VC-GAN.',
                        metavar='F_BETA', default=1, type=float)
    parser.add_argument('--cls_alpha', help='Classification hyper in VC-GAN.',
                        metavar='CLS_ALPHA', default=0, type=float)
    parser.add_argument('--epsilon_loss', help='Continuous lambda for INFO-GAN and VC-GAN.',
                        metavar='EPSILON_LOSS', default=0.4, type=float)
    parser.add_argument('--where_feat_map', help='Which layer of feat map to use for F_loss.',
                        metavar='WHERE_FEAT_MAP', default=15, type=int)
    parser.add_argument('--latent_type', help='What type of latent priori to use.',
                        metavar='LATENT_TYPE', default='uniform', choices=['uniform', 'normal', 'trunc_normal'], type=str)
    parser.add_argument( '--random_eps',
        help='If use random epsilon in vc_gan_with_vc_head loss.',
        default=False, metavar='RANDOM_EPS', type=_str_to_bool)
    parser.add_argument('--delta_type', help='What type of delta use.',
                        metavar='DELTA_TYPE', default='onedim', choices=['onedim', 'fulldim'], type=str)
    parser.add_argument('--cascading', help='If use cascading',
                        default=False, metavar='CASCADING', type=_str_to_bool)
    parser.add_argument('--connect_mode', help='How fake1 and fake2 connected.',
                        default='concat', metavar='CONNECT_MODE', type=str)
    parser.add_argument('--fmap_decay', help='fmap decay for network building.',
                        metavar='FMAP_DECAY', default=0.15, type=float)

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print('Error: dataset root directory does not exist.')
        sys.exit(1)

    if args.config_id not in _valid_configs:
        print('Error: --config value must be one of: ',
              ', '.join(_valid_configs))
        sys.exit(1)

    for metric in args.metrics:
        if metric not in metric_defaults:
            print('Error: unknown metric \'%s\'' % metric)
            sys.exit(1)

    run(**vars(args))


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
