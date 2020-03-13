#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: grab_traversals.py
# --- Creation Date: 02-03-2020
# --- Last Modified: Mon 02 Mar 2020 03:01:03 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Grab a row of traversals from generated grids.
"""

import argparse
import os
import glob
import pdb
from PIL import Image


def crop_images(args):
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    source_imgs_path = sorted(glob.glob(os.path.join(args.source_dir, 'img_*.png')))
    for source_path in source_imgs_path[:args.n_used_sources]:
        img_name = os.path.basename(source_path)
        img = Image.open(source_path)
        img_x, img_y = img.size
        y_s = (args.row - 1) * args.crop_h
        crop_row = img.crop((0, y_s, img_x, y_s + args.crop_h))
        save_path = os.path.join(args.result_dir, img_name)
        crop_row.save(save_path)


def main():
    parser = argparse.ArgumentParser(description='Project description.')
    parser.add_argument('--result_dir',
                        help='Results directory.',
                        type=str,
                        default='/mnt/hdd/repo_results/test')
    parser.add_argument('--source_dir',
                        help='Grid directory.',
                        type=str,
                        default='/mnt/hdd/Datasets/test_data')
    parser.add_argument('--row',
                        help='Which row to grab. Starting from 1.',
                        type=int)
    parser.add_argument('--crop_h', type=int, default=128)
    parser.add_argument('--crop_w', type=int, default=128)
    parser.add_argument('--n_used_sources', type=int, default=20)

    args = parser.parse_args()

    crop_images(args)


if __name__ == "__main__":
    main()
