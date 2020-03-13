# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import copy
import os
import sys

import dnnlib
from dnnlib import EasyDict

from metrics.metric_defaults import metric_defaults
from training.spatial_biased_modular_networks import split_module_names, LATENT_MODULES

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


def run(
        dataset,
        data_dir,
        result_dir,
        config_id,
        num_gpus,
        total_kimg,
        gamma,
        mirror_augment,
        metrics,
        resume_pkl,
        D_global_size=3,
        C_global_size=0,  # Global C_latents.
        sb_C_global_size=4,
        C_local_hfeat_size=0,  # Local heatmap*features learned C_latents.
        C_local_heat_size=0,  # Local heatmap learned C_latents.
        n_samples_per=10,
        module_list=None,
        single_const=True,
        model_type='spatial_biased'):
    # print('module_list:', module_list)
    train = EasyDict(
        run_func_name='training.training_loop_dsp.training_loop_dsp'
    )  # Options for training loop.
    if model_type == 'spatial_biased':
        G = EasyDict(
            func_name=
            'training.spatial_biased_networks.G_main_spatial_biased_dsp',
            mapping_fmaps=128,
            fmap_max=128,
            latent_size=D_global_size + sb_C_global_size,
            dlatent_size=D_global_size + sb_C_global_size,
            D_global_size=D_global_size,
            sb_C_global_size=sb_C_global_size
        )  # Options for generator network.
        desc = 'spatial_biased_net'
    elif model_type == 'sb_general':
        G = EasyDict(
            func_name=
            'training.spatial_biased_networks.G_main_spatial_biased_dsp',
            synthesis_func='G_synthesis_sb_general_dsp',
            mapping_fmaps=128,
            fmap_max=128,
            latent_size=D_global_size + C_global_size + sb_C_global_size +
            C_local_hfeat_size + C_local_heat_size,
            dlatent_size=D_global_size + C_global_size + sb_C_global_size +
            C_local_hfeat_size + C_local_heat_size,
            D_global_size=D_global_size,
            C_global_size=C_global_size,
            sb_C_global_size=sb_C_global_size,
            C_local_hfeat_size=C_local_hfeat_size,
            C_local_heat_size=C_local_heat_size,
            use_noise=False)  # Options for generator network.
        desc = 'sb_general_net'
    elif model_type == 'sb_modular':
        module_list = _str_to_list(module_list)
        key_ls, size_ls, count_dlatent_size, _ = split_module_names(
            module_list)
        for i, key in enumerate(key_ls):
            if key.startswith('D_global'):
                D_global_size = size_ls[i]
                break
        print('D_global_size:', D_global_size)
        G = EasyDict(
            func_name=
            'training.spatial_biased_networks.G_main_spatial_biased_dsp',
            synthesis_func='G_synthesis_sb_modular',
            mapping_fmaps=128,
            fmap_max=128,
            latent_size=count_dlatent_size,
            dlatent_size=count_dlatent_size,
            D_global_size=D_global_size,
            module_list=module_list,
            single_const=single_const,
            use_noise=False)  # Options for generator network.
        desc = 'sb_modular_net'
    elif model_type == 'sb_singlelayer_modi':
        G = EasyDict(func_name='training.simple_networks.G_main_simple_dsp',
                     synthesis_func='G_synthesis_sb_singlelayer_modi_dsp',
                     mapping_fmaps=128,
                     fmap_max=128,
                     latent_size=D_global_size + sb_C_global_size,
                     dlatent_size=D_global_size + sb_C_global_size,
                     D_global_size=D_global_size,
                     sb_C_global_size=sb_C_global_size
                     )  # Options for generator network.
        desc = 'sb_singlelayer_net'
    elif model_type == 'stylegan2':
        G = EasyDict(
            func_name=
            'training.spatial_biased_networks.G_main_spatial_biased_dsp',
            dlatent_avg_beta=None,
            mapping_fmaps=128,
            fmap_max=128,
            latent_size=12,
            D_global_size=D_global_size,
            sb_C_global_size=sb_C_global_size
        )  # Options for generator network.
        desc = 'stylegan2_net'
    elif model_type == 'simple':
        G = EasyDict(func_name='training.simple_networks.G_main_simple_dsp',
                     latent_size=D_global_size + sb_C_global_size,
                     dlatent_size=D_global_size + sb_C_global_size,
                     D_global_size=D_global_size,
                     sb_C_global_size=sb_C_global_size
                     )  # Options for generator network.
    else:
        raise ValueError('Not supported model tyle: ' + model_type)

    if model_type == 'simple':
        D = EasyDict(func_name='training.simple_networks.D_simple_dsp'
                     )  # Options for discriminator network.
    else:
        D = EasyDict(func_name='training.networks_stylegan2.D_stylegan2',
                     fmap_max=128)  # Options for discriminator network.
        # D         = EasyDict(func_name='training.spatial_biased_networks.D_with_discrete_dsp', fmap_max=128)  # Options for discriminator network.
    G_opt = EasyDict(beta1=0.0, beta2=0.99,
                     epsilon=1e-8)  # Options for generator optimizer.
    D_opt = EasyDict(beta1=0.0, beta2=0.99,
                     epsilon=1e-8)  # Options for discriminator optimizer.
    G_loss = EasyDict(
        func_name='training.loss.G_logistic_ns_dsp',
        D_global_size=D_global_size)  # Options for generator loss.
    D_loss = EasyDict(
        func_name='training.loss.D_logistic_r1_dsp',
        D_global_size=D_global_size)  # Options for discriminator loss.
    sched = EasyDict()  # Options for TrainingSchedule.
    grid = EasyDict(
        size='1080p',
        layout='random')  # Options for setup_snapshot_image_grid().
    sc = dnnlib.SubmitConfig()  # Options for dnnlib.submit_run().
    tf_config = {'rnd.np_random_seed': 1000}  # Options for tflib.init_tf().

    train.data_dir = data_dir
    train.total_kimg = total_kimg
    train.mirror_augment = mirror_augment
    train.image_snapshot_ticks = train.network_snapshot_ticks = 10
    sched.G_lrate_base = sched.D_lrate_base = 0.002
    sched.minibatch_size_base = 32
    sched.minibatch_gpu_base = 4
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
    if config_id != 'config-f':
        G.fmap_base = D.fmap_base = 8 << 10

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
    kwargs.update(G_args=G,
                  D_args=D,
                  G_opt_args=G_opt,
                  D_opt_args=D_opt,
                  G_loss_args=G_loss,
                  D_loss_args=D_loss,
                  traversal_grid=True)
    if model_type == 'sb_modular':
        n_continuous = 0
        for i, key in enumerate(key_ls):
            m_name = key.split('-')[0]
            if (m_name in LATENT_MODULES) and (not m_name == 'D_global'):
                n_continuous += size_ls[i]
    else:
        n_continuous = C_global_size + sb_C_global_size + \
            C_local_hfeat_size + C_local_heat_size
    kwargs.update(dataset_args=dataset_args,
                  sched_args=sched,
                  grid_args=grid,
                  metric_arg_list=metrics,
                  tf_config=tf_config,
                  resume_pkl=resume_pkl,
                  n_discrete=D_global_size,
                  n_continuous=n_continuous,
                  n_samples_per=n_samples_per)
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
        description='Train StyleGAN2.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--result-dir',
        help='Root directory for run results (default: %(default)s)',
        default='results',
        metavar='DIR')
    parser.add_argument('--data-dir',
                        help='Dataset root directory',
                        required=True)
    parser.add_argument('--dataset', help='Training dataset', required=True)
    parser.add_argument('--config',
                        help='Training config (default: %(default)s)',
                        default='config-e',
                        dest='config_id',
                        metavar='CONFIG')
    parser.add_argument('--num-gpus',
                        help='Number of GPUs (default: %(default)s)',
                        default=1,
                        type=int,
                        metavar='N')
    parser.add_argument(
        '--total-kimg',
        help='Training length in thousands of images (default: %(default)s)',
        metavar='KIMG',
        default=25000,
        type=int)
    parser.add_argument(
        '--gamma',
        help='R1 regularization weight (default is config dependent)',
        default=None,
        type=float)
    parser.add_argument('--mirror-augment',
                        help='Mirror augment (default: %(default)s)',
                        default=False,
                        metavar='BOOL',
                        type=_str_to_bool)
    parser.add_argument(
        '--metrics',
        help='Comma-separated list of metrics or "none" (default: %(default)s)',
        default='None',
        type=_parse_comma_sep)
    parser.add_argument('--model_type',
                        help='Type of model to train',
                        default='spatial_biased',
                        type=str,
                        metavar='MODEL_TYPE',
                        choices=[
                            'spatial_biased', 'stylegan2', 'simple',
                            'sb_singlelayer_modi', 'sb_general', 'sb_modular'
                        ])
    parser.add_argument('--resume_pkl',
                        help='Continue training using pretrained pkl.',
                        default=None,
                        metavar='RESUME_PKL',
                        type=str)
    parser.add_argument(
        '--D_global_size',
        help=
        'Number of global discrete latents in model (default: %(default)s)',
        metavar='N_GLOBAL_DISCRETE',
        default=3,
        type=int)
    parser.add_argument(
        '--C_global_size',
        help=
        'Number of global continuous latents in model (default: %(default)s)',
        metavar='N_GLOBAL_CONTINUOUS',
        default=0,
        type=int)
    parser.add_argument(
        '--sb_C_global_size',
        help=
        'Number of spatial-biased continuous latents in model (default: %(default)s)',
        metavar='N_GLOBAL_C_SPATIAL_BIASED',
        default=4,
        type=int)
    parser.add_argument(
        '--C_local_hfeat_size',
        help=
        'Number of Local heatmap*features learned continuous latents in model (default: %(default)s)',
        metavar='N_LOCAL_C_HFEAT',
        default=0,
        type=int)
    parser.add_argument(
        '--C_local_heat_size',
        help=
        'Number of Local heatmap learned continuous latents in model (default: %(default)s)',
        metavar='N_LOCAL_C_HEAT',
        default=0,
        type=int)
    parser.add_argument(
        '--n_samples_per',
        help=
        'Number of samples for each line in traversal (default: %(default)s)',
        metavar='N_SHOWN_SAMPLES_PER_LINE',
        default=10,
        type=int)
    parser.add_argument('--module_list',
                        help='Module list for modular network.',
                        default=None,
                        metavar='MODULE_LIST',
                        type=str)
    parser.add_argument(
        '--single_const',
        help=
        'Use a single constant feature at the top or not (if not, n_classes of const feature maps will be used and gathered).',
        default=True,
        metavar='BOOL',
        type=_str_to_bool)

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
