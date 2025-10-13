gpu_id=0

for seed in 42
do
    echo "Running CUB with seed $seed on GPU $gpu_id"
    CUDA_VISIBLE_DEVICES=$gpu_id python sae_main.py \
                --seed $seed \
                --dataset cub \
                --base_model_name clip_vit-b-32 \
                --lr 3e-4 \
                --epochs 20 \
                --batch_size 256 \
                --save_model_dir ./weights/

    sleep 30
done