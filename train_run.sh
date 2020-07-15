python run_training_vc.py \
    --result-dir /mnt/hdd/repo_results/stylegan2vp/results_vc \
    --data-dir /mnt/hdd/Datasets/CelebA_dataset \
    --dataset celeba_tfr \
    --num-gpus 2 \
    --model_type vc_gan_with_vc_head \
    --C_lambda 0.01 \
    --random_eps True \
    --latent_type uniform \
    --delta_type onedim \
    --module_list '[Conv-up-1, C_global-10, Conv-id-2, Noise-2, Conv-up-1, C_global-10, Conv-id-1, Noise-2, Conv-up-1, C_global-5, Conv-id-2, Noise-2, Conv-id-1, Noise-2, Conv-up-1, C_global-5, Conv-id-1, Noise-2, Conv-id-2, Conv-up-1, Conv-id-1]'

#python run_training_vc.py \
    #--result-dir /mnt/hdd/repo_results/stylegan2vp/results_info \
    #--data-dir /mnt/hdd/Datasets/CelebA_dataset \
    #--dataset celeba_tfr \
    #--num-gpus 2 \
    #--model_type info_gan \
    #--C_lambda 0.01 \
    #--latent_type uniform \
    #--module_list '[Conv-up-1, C_global-10, Conv-id-2, Noise-2, Conv-up-1, C_global-10, Conv-id-1, Noise-2, Conv-up-1, C_global-5, Conv-id-2, Noise-2, Conv-id-1, Noise-2, Conv-up-1, C_global-5, Conv-id-1, Noise-2, Conv-id-2, Conv-up-1, Conv-id-1]'
