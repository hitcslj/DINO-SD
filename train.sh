CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 36000 run.py \
    --model_name nusc \
    --num_workers 4 \
    --config configs/nusc.txt \
    --log_dir ./log_baseline