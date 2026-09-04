#!/usr/bin/env bash
# =============================================================================
# run_all_unified.sh
# Fine-tune CLIP, DINO, and ALIGN with LoRA and DoRA on all 16 datasets.
#
# 16 datasets × 3 models × 2 methods = 96 training runs.
#
# Usage:
#   bash run_all_unified.sh              # all 96 runs
#   bash run_all_unified.sh clip lora    # only CLIP-LoRA runs
#   bash run_all_unified.sh dino dora    # only DINO-DoRA runs
#
# Missing-data check: if a dataset root directory does not exist the run is
# skipped with a warning rather than crashing the entire sweep.
# =============================================================================

set -uo pipefail   # removed -e so missing datasets don't abort the sweep

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/home/sunayana/Documents/Concept_LoRA
DATA=$REPO/data
SAVE=$REPO/unified_weights
LOG_DIR=$REPO/unified_logs

mkdir -p "$LOG_DIR" "$SAVE"

# ── Filter: pass model and/or method as positional args to limit scope ─────
FILTER_MODEL="${1:-all}"   # clip | dino | align | all
FILTER_METHOD="${2:-all}"  # lora | dora | all

# ── Common hyper-parameters ───────────────────────────────────────────────────
BACKBONE="ViT-B/16"
SHOTS=16
N_ITERS=100
LR=2e-4
RANK=4
ALPHA=1
POSITION=all
ENCODER=both          # only used by CLIP; ignored by DINO/ALIGN
PARAMS="q k v"
NO_LP="--no_linear_probe"  # set to "" to enable the linear-probe baseline for DINO
BATCH=32

cd "$REPO"

# ── Helper: check directory exists, then run ───────────────────────────────────
run() {
    local model=$1
    local method=$2
    local dataset=$3
    local root=$4
    local check_dir="${5:-$root}"   # optional 5th arg: dir to check for existence

    # Skip if filtered
    [[ "$FILTER_MODEL" != "all" && "$FILTER_MODEL" != "$model" ]] && return 0
    [[ "$FILTER_METHOD" != "all" && "$FILTER_METHOD" != "$method" ]] && return 0

    # Skip if dataset data directory is missing
    if [[ ! -d "$check_dir" ]]; then
        echo ""
        echo "SKIP $model/$method/$dataset — data directory not found: $check_dir"
        return 0
    fi

    local log="$LOG_DIR/${model}_${method}_${dataset}.log"
    echo ""
    echo "============================================================"
    echo " Model: $model  |  Method: $method  |  Dataset: $dataset"
    echo "============================================================"

    python unified_finetune.py \
        --model      "$model"    \
        --method     "$method"   \
        --dataset    "$dataset"  \
        --root_path  "$root"     \
        --backbone   "$BACKBONE" \
        --shots      "$SHOTS"    \
        --n_iters    "$N_ITERS"  \
        --lr         "$LR"       \
        --r          "$RANK"     \
        --alpha      "$ALPHA"    \
        --position   "$POSITION" \
        --encoder    "$ENCODER"  \
        --params     $PARAMS     \
        --batch_size "$BATCH"    \
        --save_path  "$SAVE"     \
        --filename   "adapter_weights" \
        $NO_LP \
        2>&1 | tee "$log"
}

# =============================================================================
# Dataset definitions: run <model> <method> <dataset_key> <root_path> [check_dir]
#
# check_dir (5th arg) is the directory that must exist for the run to proceed.
# If omitted it defaults to root_path itself.
# =============================================================================

for MODEL in clip dino align; do
  for METHOD in lora dora; do

    # ── Standard datasets (root = $DATA, data subdir checked) ───────────────
    run "$MODEL" "$METHOD" ucf101         "$DATA" "$DATA/UCF101"
    run "$MODEL" "$METHOD" stanford_cars  "$DATA" "$DATA/stanford_cars"
    run "$MODEL" "$METHOD" oxford_pets    "$DATA" "$DATA/oxford_pets_imagefolder"
    run "$MODEL" "$METHOD" food101        "$DATA" "$DATA/food101"
    run "$MODEL" "$METHOD" oxford_flowers "$DATA" "$DATA/flowers102_imagefolder"
    run "$MODEL" "$METHOD" fgvc           "$DATA" "$DATA/fgvc_imagefolder"
    run "$MODEL" "$METHOD" eurosat        "$DATA" "$DATA/eurosat"
    run "$MODEL" "$METHOD" dtd            "$DATA" "$DATA/dtd"

    # ── ImageNet robustness variants (root = $DATA) ─────────────────────────
    run "$MODEL" "$METHOD" imagenet_v2     "$DATA" "$DATA/imagenetv2"
    run "$MODEL" "$METHOD" imagenet_sketch "$DATA" "$DATA/sketch"
    run "$MODEL" "$METHOD" imagenet_a      "$DATA" "$DATA/imagenet-a"
    run "$MODEL" "$METHOD" imagenet_r      "$DATA" "$DATA/imagenet-r"

    # ── Datasets with their own subdirectory ────────────────────────────────
    run "$MODEL" "$METHOD" caltech101  "$DATA/caltech-101" "$DATA/caltech-101/Caltech101"
    run "$MODEL" "$METHOD" sun397      "$DATA"             "$HOME/.cache/huggingface/hub/datasets--1aurent--SUN397"
    run "$MODEL" "$METHOD" medmnist    "$DATA/pathmnist_imagefolder" "$DATA/pathmnist_imagefolder"
    run "$MODEL" "$METHOD" chexpert    "$DATA/chexpert"   "$DATA/chexpert"

  done
done

echo ""
echo "============================================================"
echo " All runs complete."
echo " Weights : $SAVE"
echo " Logs    : $LOG_DIR"
echo "============================================================"
