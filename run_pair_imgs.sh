python run_pair_generator_vc.py \
    --network_pkl /mnt/hdd/repo_results/stylegan2vp/test/test.pkl \
    --n_imgs 10000 \
    --n_discrete 0 \
    --n_continuous 30 \
    --batch_size 100 \
    --latent_type onedim \
    --model_type vc_gan_with_vc_head \
    --result-dir /mnt/hdd/repo_results/stylegan2vp/test/pair_dataset
