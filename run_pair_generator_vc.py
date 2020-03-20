#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: run_pair_generator_vc.py
# --- Creation Date: 27-02-2020
# --- Last Modified: Fri 20 Mar 2020 15:48:45 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Generate a image-pair dataset
"""

import argparse
import numpy as np
from PIL import Image
import dnnlib
import dnnlib.tflib as tflib
import re
import os
import sys

import pretrained_networks
from training import misc
from training.training_loop_dsp import get_grid_latents

#----------------------------------------------------------------------------


def generate_image_pairs(network_pkl,
                         n_imgs,
                         model_type,
                         n_discrete,
                         n_continuous,
                         result_dir,
                         batch_size=10,
                         latent_type='onedim'):
    print('Loading networks from "%s"...' % network_pkl)
    tflib.init_tf()
    if (model_type == 'info_gan') or (model_type == 'vc_gan_with_vc_head'):
        _G, _D, I, Gs = misc.load_pkl(network_pkl)
    else:
        _G, _D, Gs = misc.load_pkl(network_pkl)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # _G, _D, Gs = pretrained_networks.load_networks(network_pkl)

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.randomize_noise = False

    n_batches = n_imgs // batch_size

    for i in range(n_batches):
        print('Generating image pairs %d/%d ...' % (i, n_batches))
        grid_labels = np.zeros([batch_size, 0], dtype=np.float32)

        if n_discrete > 0:
            cat_dim = np.random.randint(0, n_discrete, size=[batch_size])
            cat_onehot = np.zeros((batch_size, n_discrete))
            cat_onehot[np.arange(cat_dim.size), cat_dim] = 1

        z_1 = np.random.uniform(low=-2,
                                high=2,
                                size=[batch_size, n_continuous])
        z_2 = np.random.uniform(low=-2,
                                high=2,
                                size=[batch_size, n_continuous])
        if latent_type == 'onedim':
            delta_dim = np.random.randint(0, n_continuous, size=[batch_size])
            delta_onehot = np.zeros((batch_size, n_continuous))
            delta_onehot[np.arange(delta_dim.size), delta_dim] = 1
            z_2 = np.where(delta_onehot > 0, z_2, z_1)
        delta_z = z_1 - z_2

        if i == 0:
            labels = delta_z
        else:
            labels = np.concatenate([labels, delta_z], axis=0)

        if n_discrete > 0:
            z_1 = np.concatenate((cat_onehot, z_1), axis=1)
            z_2 = np.concatenate((cat_onehot, z_2), axis=1)

        fakes_1 = Gs.run(z_1,
                            grid_labels,
                            is_validation=True,
                            minibatch_size=batch_size,
                            **Gs_kwargs)
        fakes_2 = Gs.run(z_2,
                            grid_labels,
                            is_validation=True,
                            minibatch_size=batch_size,
                            **Gs_kwargs)
        print('fakes_1.shape:', fakes_1.shape)
        print('fakes_2.shape:', fakes_2.shape)

        for j in range(fakes_1.shape[0]):
            pair_np = np.concatenate([fakes_1[j], fakes_2[j]], axis=2)
            img = misc.convert_to_pil_image(pair_np, [-1, 1])
            # pair_np = (pair_np * 255).astype(np.uint8)
            # img = Image.fromarray(pair_np)
            img.save(
                os.path.join(result_dir,
                             'pair_%06d.jpg' % (i * batch_size + j)))
    np.save(os.path.join(result_dir, 'labels.npy'), labels)


#----------------------------------------------------------------------------

_examples = '''examples:

  # Generate image pairs
  python %(prog)s --network_pkl=results/info_gan.pkl --n_imgs=5 --result_dir ./results
'''


#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='VC-GAN and INFO-GAN image-pair generator.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--network_pkl',
                        help='Network pickle filename',
                        required=True)
    parser.add_argument('--n_imgs',
                        type=int,
                        help='Number of image pairs to generate',
                        required=True)
    parser.add_argument('--n_discrete',
                        type=int,
                        help='Number of discrete latents',
                        default=0)
    parser.add_argument('--n_continuous',
                        type=int,
                        help='Number of continuous latents',
                        default=14)
    parser.add_argument('--batch_size',
                        type=int,
                        help='Batch size for generation',
                        default=10)
    parser.add_argument('--latent_type',
                        type=str,
                        help='What type of latent difference to use',
                        default='onedim',
                        choices=['onedim', 'fulldim'])
    parser.add_argument('--model_type',
                        type=str,
                        help='Which model is this pkl',
                        default='vc_gan_with_vc_head',
                        choices=['info_gan', 'vc_gan', 'vc_gan_with_vc_head'])
    parser.add_argument('--result-dir',
                        help='Root directory to store this dataset',
                        required=True,
                        metavar='DIR')

    args = parser.parse_args()
    kwargs = vars(args)

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs['result_dir']

    dnnlib.submit_run(sc, 'run_pair_generator_vc.generate_image_pairs',
                      **kwargs)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
