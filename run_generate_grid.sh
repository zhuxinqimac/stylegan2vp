CUDA_VISIBLE_DEVICES=0 \
    python run_generator_vc.py \
    --network_pkl /mnt/hdd/repo_results/stylegan2vp/test/test.pkl \
    --n_imgs 100 \
    --n_discrete 0 \
    --n_continuous 30 \
    --model_type vc_gan_with_vc_head \
    --result-dir /mnt/hdd/repo_results/stylegan2vp/test/generation
