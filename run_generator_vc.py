#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: run_generator_vc.py
# --- Creation Date: 08-02-2020
# --- Last Modified: Mon 02 Mar 2020 14:27:10 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Generator script for vc-gan and info-gan models.
"""

import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys

import pretrained_networks
from training import misc
from training.training_loop_dsp import get_grid_latents

#----------------------------------------------------------------------------


def generate_images(network_pkl,
                    n_imgs,
                    model_type,
                    n_discrete,
                    n_continuous,
                    n_samples_per=10):
    print('Loading networks from "%s"...' % network_pkl)
    tflib.init_tf()
    if (model_type == 'info_gan') or (model_type == 'vc_gan_with_vc_head'):
        _G, _D, I, Gs = misc.load_pkl(network_pkl)
    else:
        _G, _D, Gs = misc.load_pkl(network_pkl)

    # _G, _D, Gs = pretrained_networks.load_networks(network_pkl)

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8,
                                      nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False

    for idx in range(n_imgs):
        print('Generating image %d/%d ...' % (idx, n_imgs))

        if n_discrete == 0:
            grid_labels = np.zeros([n_continuous * n_samples_per, 0],
                                   dtype=np.float32)
        else:
            grid_labels = np.zeros(
                [n_discrete * n_continuous * n_samples_per, 0],
                dtype=np.float32)

        grid_size, grid_latents, grid_labels = get_grid_latents(
            n_discrete, n_continuous, n_samples_per, _G, grid_labels)
        grid_fakes, _ = Gs.run(grid_latents,
                            grid_labels,
                            is_validation=True,
                            minibatch_size=4,
                            randomize_noise=False)
        misc.save_image_grid(grid_fakes,
                             dnnlib.make_run_dir_path('img_%04d.png' % idx),
                             drange=[-1, 1],
                             grid_size=grid_size)

        # z = rnd.randn(1, *Gs.input_shapes[0][1:]) # [minibatch, component]
        # z_tiled = np.tile(z, (100, 1))
        # for dim in range(10):
        # z_tiled[dim*10:dim*10+10, dim] = np.arange(-2, 2, 0.4)

        # z_cls = np.zeros((1, *Gs.input_shapes[1][1:]))
        # z_cls[0, rnd.randint(z_cls.shape[1])] = 1
        # z_cls_tiled = np.tile(z_cls, (100, 1))

        # tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        # # images = Gs.run(z, z_cls, **Gs_kwargs)[:, :, :, 0] # [minibatch, height, width, channel]
        # images = Gs.run(z_tiled, z_cls_tiled, **Gs_kwargs)[:, :, :, 0] # [minibatch, height, width, channel]
        # # PIL.Image.fromarray(images[0], 'L').save(dnnlib.make_run_dir_path('seed%04d.png' % seed))
        # misc.save_image_grid(images, dnnlib.make_run_dir_path('seed%04d.png' % seed), drange=[0, 255], grid_size=(10, 10))


#----------------------------------------------------------------------------

_examples = '''examples:

  # Generate images traversals
  python %(prog)s --network_pkl=results/info_gan.pkl --n_imgs=5 --result_dir ./results
'''


#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='''VC-GAN and INFO-GAN generator.

Run 'python %(prog)s --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--network_pkl',
                        help='Network pickle filename',
                        required=True)
    parser.add_argument('--n_imgs',
                        type=int,
                        help='Number of images to generate',
                        required=True)
    parser.add_argument('--n_discrete',
                        type=int,
                        help='Number of discrete latents',
                        default=0)
    parser.add_argument('--n_continuous',
                        type=int,
                        help='Number of continuous latents',
                        default=14)
    parser.add_argument('--n_samples_per',
                        type=int,
                        help='Number of samples per row',
                        default=10)
    parser.add_argument('--model_type',
                        type=str,
                        help='Which model is this pkl',
                        default='vc_gan_with_vc_head',
                        choices=['info_gan', 'vc_gan', 'vc_gan_with_vc_head'])
    parser.add_argument(
        '--result-dir',
        help='Root directory for run results (default: %(default)s)',
        default='results',
        metavar='DIR')

    args = parser.parse_args()
    kwargs = vars(args)

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')

    dnnlib.submit_run(sc, 'run_generator_vc.generate_images', **kwargs)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
