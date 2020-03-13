#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: run_unsupervised_acc.py
# --- Creation Date: 12-02-2020
# --- Last Modified: Thu 13 Feb 2020 04:04:47 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Calculate the unsupervised classification accuracy 
of Variation Consistency Models
"""

import argparse
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import re
import pdb
import sys

import projector_vc
import pretrained_networks
from training import dataset
from training import misc

#----------------------------------------------------------------------------

def project_image(proj, targets, png_prefix, num_snapshots):
    snapshot_steps = set(proj.num_steps - np.linspace(0, proj.num_steps, num_snapshots, endpoint=False, dtype=int))
    misc.save_image_grid(targets, png_prefix + 'target.png', drange=[-1,1])
    proj.start(targets)
    while proj.get_cur_step() < proj.num_steps:
        print('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step()
        if proj.get_cur_step() in snapshot_steps:
            misc.save_image_grid(proj.get_images(), png_prefix + 'step%04d.png' % proj.get_cur_step(), drange=[-1,1])
    # print(proj.get_predictions())
    # print('\r%-30s\r' % '', end='', flush=True)
    return proj.get_predictions()

#----------------------------------------------------------------------------

def project_generated_images(network_pkl, seeds, num_snapshots, truncation_psi, 
                             D_size=0, minibatch_size=1, use_VGG=True):
    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, I, Gs = misc.load_pkl(network_pkl)
    # _G, _D, Gs = misc.load_pkl(network_pkl)
    # _G, _D, Gs = pretrained_networks.load_networks(network_pkl)

    proj = projector_vc.ProjectorVC()
    proj.set_network(Gs, minibatch_size=minibatch_size, D_size=D_size, use_VGG=use_VGG, num_steps=num_steps)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.truncation_psi = truncation_psi

    for seed_idx, seed in enumerate(seeds):
        print('Projecting seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:])
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
        images = Gs.run(z, None, **Gs_kwargs)
        project_image(proj, targets=images, png_prefix=dnnlib.make_run_dir_path('seed%04d-' % seed), num_snapshots=num_snapshots)

#----------------------------------------------------------------------------

def project_real_images(network_pkl, dataset_name, data_dir, num_images, num_snapshots, 
                        D_size=0, minibatch_size=1, use_VGG=True):
    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, I, Gs = misc.load_pkl(network_pkl)
    # _G, _D, Gs = misc.load_pkl(network_pkl)
    # _G, _D, Gs = pretrained_networks.load_networks(network_pkl)

    proj = projector_vc.ProjectorVC()
    proj.set_network(Gs, minibatch_size=minibatch_size, D_size=D_size, use_VGG=use_VGG, num_steps=num_steps)

    print('Loading images from "%s"...' % dataset_name)
    dataset_obj = dataset.load_dataset(data_dir=data_dir, tfrecord_dir=dataset_name, max_label_size='full', repeat=False, shuffle_mb=0)
    assert dataset_obj.shape == Gs.output_shape[1:]

    for image_idx in range(num_images):
        print('Projecting image %d/%d ...' % (image_idx, num_images))
        images, _labels = dataset_obj.get_minibatch_np(minibatch_size)
        print('images.shape:', images.shape)
        print('_labels.shape:', _labels.shape)
        print('_labels:', _labels)
        print('argmax of _labels:', np.argmax(_labels, axis=1))
        # pdb.set_trace()
        images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
        project_image(proj, targets=images, png_prefix=dnnlib.make_run_dir_path('image%04d-' % image_idx), num_snapshots=num_snapshots)
#----------------------------------------------------------------------------

def classify_images(network_pkl, train_dataset_name, data_dir, n_batches_of_train_imgs, 
                    test_dataset_name=None, D_size=0, minibatch_size=1, use_VGG=True, log_freq=10, num_steps=200):
    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, I, Gs = misc.load_pkl(network_pkl)
    # _G, _D, Gs = misc.load_pkl(network_pkl)
    # _G, _D, Gs = pretrained_networks.load_networks(network_pkl)

    proj = projector_vc.ProjectorVC()
    proj.set_network(Gs, minibatch_size=minibatch_size, D_size=D_size, use_VGG=use_VGG, num_steps=num_steps)

    print('Loading images from "%s"...' % train_dataset_name)
    dataset_obj = dataset.load_dataset(data_dir=data_dir, tfrecord_dir=train_dataset_name, max_label_size='full', repeat=False, shuffle_mb=0)
    assert dataset_obj.shape == Gs.output_shape[1:]

    vote_matrix = np.zeros((D_size, D_size), dtype=np.int32)
    # Training
    all_correct_train = 0
    all_preds_train = 0
    for image_idx in range(n_batches_of_train_imgs):
        images, _labels = dataset_obj.get_minibatch_np(minibatch_size)
        images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
        preds = project_image(proj, targets=images, png_prefix=dnnlib.make_run_dir_path('image%04d-' % image_idx), num_snapshots=0)
        labels = np.argmax(_labels, axis=1)
        for i in range(len(preds)):
            vote_matrix[preds[i], labels[i]] += 1
        pred_to_label = np.argmax(vote_matrix, axis=1)

        # Calc training acc
        preds_l = pred_to_label[preds]
        all_preds_train += len(preds_l)
        all_correct_train += np.sum(preds_l == labels)
        if image_idx % log_freq == 0:
            print('Training Acc: ', float(all_correct_train) / float(all_preds_train))

    print('Loading images from "%s"...' % test_dataset_name)
    dataset_obj = dataset.load_dataset(data_dir=data_dir, tfrecord_dir=test_dataset_name, max_label_size='full', repeat=False, shuffle_mb=0)
    print('Whole testing set size: ', dataset_obj.label_size)
    # pdb.set_trace()
    assert dataset_obj.shape == Gs.output_shape[1:]
    all_correct = 0
    all_preds = 0
    for image_idx in range(10000 // minibatch_size):
        images, _labels = dataset_obj.get_minibatch_np(minibatch_size)
        images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
        preds = project_image(proj, targets=images, png_prefix=dnnlib.make_run_dir_path('image%04d-' % image_idx), num_snapshots=0)
        preds_l = pred_to_label[preds]
        labels = np.argmax(_labels, axis=1)
        all_preds += len(preds_l)
        all_correct += np.sum(preds_l == labels)
        if image_idx % log_freq == 0:
            print('Testing Acc: ', float(all_correct) / float(all_preds))

#----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return range(int(m.group(1)), int(m.group(2))+1)
    vals = s.split(',')
    return [int(x) for x in vals]

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

#----------------------------------------------------------------------------

_examples = '''examples:

  # Project generated images
  python %(prog)s project-generated-images --network=gdrive:networks/stylegan2-car-config-f.pkl --seeds=0,1,5

  # Project real images
  python %(prog)s project-real-images --network=gdrive:networks/stylegan2-car-config-f.pkl --dataset=car --data-dir=~/datasets

'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''VC-Gan Classifier.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    project_generated_images_parser = subparsers.add_parser('project-generated-images', help='Project generated images')
    project_generated_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    project_generated_images_parser.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', default=range(3))
    project_generated_images_parser.add_argument('--num-snapshots', type=int, help='Number of snapshots (default: %(default)s)', default=5)
    project_generated_images_parser.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=1.0)
    project_generated_images_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    project_generated_images_parser.add_argument('--D_size', type=int, help='Number of discrete latents', default=10)
    project_generated_images_parser.add_argument('--minibatch_size', type=int, help='Minibatch size', default=1)

    project_real_images_parser = subparsers.add_parser('project-real-images', help='Project real images')
    project_real_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    project_real_images_parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    project_real_images_parser.add_argument('--dataset', help='Training dataset', dest='dataset_name', required=True)
    project_real_images_parser.add_argument('--num-snapshots', type=int, help='Number of snapshots (default: %(default)s)', default=5)
    project_real_images_parser.add_argument('--num-images', type=int, help='Number of images to project (default: %(default)s)', default=3)
    project_real_images_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    project_real_images_parser.add_argument('--D_size', type=int, help='Number of discrete latents', default=10)
    project_real_images_parser.add_argument('--minibatch_size', type=int, help='Minibatch size', default=1)
    project_real_images_parser.add_argument('--use_VGG', help='If use VGG for distance eval', default=True, metavar='BOOL', type=_str_to_bool)

    classify_real_images_parser = subparsers.add_parser('classify-real-images', help='Project real images')
    classify_real_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    classify_real_images_parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    classify_real_images_parser.add_argument('--train_dataset', help='Training dataset', dest='train_dataset_name', required=True)
    classify_real_images_parser.add_argument('--test_dataset', help='Testing dataset', dest='test_dataset_name', required=True)
    classify_real_images_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    classify_real_images_parser.add_argument('--D_size', type=int, help='Number of discrete latents', default=10)
    classify_real_images_parser.add_argument('--minibatch_size', type=int, help='Minibatch size', default=1)
    classify_real_images_parser.add_argument('--use_VGG', help='If use VGG for distance eval', default=True, metavar='BOOL', type=_str_to_bool)
    classify_real_images_parser.add_argument('--n_batches_of_train_imgs', type=int, help='Number of batches for training', default=4000)
    classify_real_images_parser.add_argument('--log_freq', type=int, help='Frequency for show acc during training', default=200)
    classify_real_images_parser.add_argument('--num_steps', type=int, help='Number of steps for inference', default=200)


    args = parser.parse_args()
    subcmd = args.command
    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    kwargs = vars(args)
    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = kwargs.pop('command')

    func_name_map = {
        'project-generated-images': 'run_unsupervised_acc.project_generated_images',
        'project-real-images': 'run_unsupervised_acc.project_real_images',
        'classify-real-images': 'run_unsupervised_acc.classify_images'
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
