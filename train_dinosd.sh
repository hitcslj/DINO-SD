CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 36000 run_dinosd.py \
    --model_name nusc\
    --num_workers 4 \
    --config configs/nusc.txt \
    --log_dir './logs_dinosd' \
    --learning_rate_pose 1e-4 \
    --learning_rate_encoder 5e-6 \
    --learning_rate_decoder 5e-5