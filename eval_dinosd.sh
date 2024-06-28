NUM_GPU=2
TYPE='robodrive'
CKPT='./weights'
CORRUPTIONS=(
        'brightness' 'dark' 'fog' 'frost' 'snow' 'contrast'
        'defocus_blur' 'glass_blur' 'motion_blur' 'zoom_blur' 'elastic_transform' 'color_quant'
        'gaussian_noise' 'impulse_noise' 'shot_noise' 'iso_noise' 'pixelate' 'jpeg_compression'
)
for CORRUPTION in "${CORRUPTIONS[@]}"
do
    python -m torch.distributed.launch --nproc_per_node ${NUM_GPU} corruption_run_dinosd.py \
        --model_name test \
        --config configs/${TYPE}.txt \
        --models_to_load depth encoder \
        --load_weights_folder=${CKPT} \
        --eval_only \
        --corruption_root '../robodrive-phase2' \
        --denoise equalize universal_denoise\
        --corruption ${CORRUPTION} \
        --phase2
done
