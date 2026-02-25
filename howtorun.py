nohup python3 train_phase1.py \
    --data_dir ./data/DeepRadar \
    --out_dir ./runs/phase1 \
    --epochs 200 \
    --batch 128 \
    --latent 100 \
    --emb_dim 32 \
    --n_critic 5 \
    --lr 1e-4 \
    --gp_lambda 10 \
    --num_workers 4 \
> logs/phase1_v2.log 2>&1 &