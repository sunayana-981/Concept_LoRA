#!/bin/bash
# Run DoRA fine-tuning on all available datasets.
# Datasets that share root=/data/: eurosat, dtd, fgvc, oxford_pets, ucf101
# Datasets with their own root:    caltech101, cub2002011, medmnist

set -e

DATA=/home/sunayana/Documents/Concept_LoRA/data
SAVE=/home/sunayana/Documents/Concept_LoRA/dora_weights
LOG_DIR=/home/sunayana/Documents/Concept_LoRA/dora_logs

mkdir -p "$LOG_DIR"

COMMON="--backbone ViT-B/16 --encoder both --params q k v --r 4 --alpha 1 --shots 16 --n_iters 100 --save_path $SAVE"

run_dataset() {
    local dataset=$1
    local root=$2
    echo ""
    echo "============================================================"
    echo " Dataset: $dataset  |  Root: $root"
    echo "============================================================"
    python -m CLIP_LoRA.dora_finetune \
        --dataset "$dataset" \
        --root_path "$root" \
        $COMMON \
        2>&1 | tee "$LOG_DIR/${dataset}.log"
}

cd /home/sunayana/Documents/Concept_LoRA

# Datasets where root_path is the parent data dir
run_dataset eurosat      "$DATA"
run_dataset dtd          "$DATA"
run_dataset fgvc         "$DATA"
run_dataset oxford_pets  "$DATA"
run_dataset ucf101       "$DATA"

# Datasets with their own dedicated root
run_dataset caltech101   "$DATA/caltech-101"
run_dataset cub2002011   "$DATA/cub2002011"
run_dataset medmnist     "$DATA/medmnist"

echo ""
echo "============================================================"
echo " All datasets complete. Weights saved to: $SAVE"
echo " Logs saved to: $LOG_DIR"
echo "============================================================"
