gpu_id=0

for seed in 42
do
    echo "Running MSCOCO with seed $seed on GPU $gpu_id"
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
                --seed $seed \
                --dataset mscoco \
                --base_model_name clip_vit-b-32 \
                --lr 3e-4 \
                --epochs 20 \
                --batch_size 256 \
                --save_model_dir ./saved_models/

    sleep 30
done