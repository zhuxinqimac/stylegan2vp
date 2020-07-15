# Learning Disentangled Representations with Latent Variation Predictability

This repository contains the code for [Learning Disentangled Representations with Latent Variation Predictability].

## Requirements

* 64-bit Python 3.6 installation. We recommend Anaconda3 with numpy 1.14.3 or newer.
* TensorFlow 1.14 or 1.15 with GPU support. The code does not support TensorFlow 2.0.

This project is based on StyleGAN2, which relies on custom TensorFlow ops that are compiled on the fly using [NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html).
To test that your NVCC installation is working correctly, run:

```.bash
nvcc test_nvcc.cu -o test_nvcc -run
| CPU says hello.
| GPU says hello.
```
For more detailed instruction about StyleGAN2 environment setup, see [StyleGAN2](https://github.com/NVlabs/stylegan2).

## CelebA Dataset
To prepare the tfrecord version of CelebA dataset, first download the original aligned-and-cropped version
from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html, then use the following code to
create tfrecord dataset:

```
python dataset_tool.py create_celeba /path/to/new_tfr_dir /path/to/downloaded_celeba_dir
```

## Training

To train a model on CelebA dataset, run this command:

```
sh train_run.sh
```

You can modify this script to train different model variants.
Note that for flag --data-dir you need to enter the parent directory path of
the actual dataset, and use --dataset for the actual dataset directory name.

## Evaluation

To evaluate trained models by variation predictability metric, run:

```
sh run_pair_imgs.sh
```

to generate a dataset of image pairs. You need to modify this script to
fit your result-dir and the trained network pkl.

Then use this repository https://github.com/zhuxinqimac/VP-metric-pytorch to
get the VP score using the generated dataset.
You should run multiple times (e.g. 3)
of this evaluation procedure to obtain an averaged score for your model.
