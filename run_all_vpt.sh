#!/usr/bin/env bash
# =============================================================================
# run_all_vpt.sh
# Visual Prompt Tuning (VPT-Deep) on CLIP, DINOv2, and SigLIP2 across all 16
# datasets used by run_all_unified.sh's LoRA/DoRA matrix. ALIGN is excluded
# (its EfficientNet vision tower has no patch-token sequence for VPT).
#
# 16 datasets x 3 models = 48 runs.
#
# Usage:
#   bash run_all_vpt.sh              # all 48 runs
#   bash run_all_vpt.sh clip         # only CLIP VPT runs
#
# Idempotent: skips a (model, dataset) pair if
# unified_weights/<model>_vpt/<dataset>/<shots>shots/seed<seed>/vpt_weights.pt
# already exists. Missing dataset directories are skipped with a warning
# rather than aborting the whole sweep.
# =============================================================================

set -uo pipefail

# Resolve relative to this script's own location rather than hardcoding a
# machine-specific absolute path -- the same repo checks out under
# different absolute paths locally vs. on a remote cluster (e.g. Turing,
# where the previous hardcoded local path caused every dataset lookup to
# resolve to a nonexistent directory and silently skip all 48 runs).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA=$REPO/data
SAVE=$REPO/unified_weights
LOG_DIR=$REPO/unified_logs_vpt

mkdir -p "$LOG_DIR"
cd "$REPO"

FILTER_MODEL="${1:-all}"   # clip | dino | siglip2 | all

SHOTS=16
SEED=1
# unified_finetune.py's LoRA/DoRA convention is n_iters=100 (total_steps =
# n_iters*shots); reduced here to keep this initial 48-run sweep to a few
# hours on a single shared GPU. Override via N_ITERS_OVERRIDE env var, e.g.
# N_ITERS_OVERRIDE=100 bash run_all_vpt.sh, for a full-quality sweep once
# these are validated.
N_ITERS="${N_ITERS_OVERRIDE:-10}"
N_PROMPT_TOKENS=10
BATCH=32

# This is a shared GPU -- another user's job can spike memory usage mid-run
# and OOM-kill us even if there was headroom at launch. train_vpt.py has no
# crash-safety-net checkpoint save (unlike the SAE trainers), so a crash
# just means no output file -- idempotency already handles that correctly
# on a retry -- but a headroom check + backoff still meaningfully reduces
# how often that happens.
MIN_FREE_MIB=4000
wait_for_gpu_headroom() {
    while true; do
        local used total free
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        free=$((total - used))
        if [ "$free" -ge "$MIN_FREE_MIB" ]; then
            return 0
        fi
        echo "[gpu-check] only ${free} MiB free (of ${total}), waiting for ${MIN_FREE_MIB}+ ..."
        sleep 30
    done
}

run() {
    local model=$1
    local dataset=$2
    local root=$3
    local check_dir="${4:-$root}"

    [[ "$FILTER_MODEL" != "all" && "$FILTER_MODEL" != "$model" ]] && return 0

    if [[ ! -d "$check_dir" ]]; then
        echo "SKIP $model/vpt/$dataset — data directory not found: $check_dir"
        return 0
    fi

    local out="$SAVE/${model}_vpt/$dataset/${SHOTS}shots/seed${SEED}/vpt_weights.pt"
    if [[ -f "$out" ]]; then
        echo "SKIP $model/vpt/$dataset — already done: $out"
        return 0
    fi

    local log="$LOG_DIR/${model}_vpt_${dataset}.log"
    local attempt=1 max_attempts=3
    while [ "$attempt" -le "$max_attempts" ]; do
        wait_for_gpu_headroom
        echo ""
        echo "============================================================"
        echo " VPT  |  Model: $model  |  Dataset: $dataset  |  Attempt ${attempt}/${max_attempts}"
        echo "============================================================"

        python3 train_vpt.py \
            --model           "$model"   \
            --dataset         "$dataset" \
            --root_path       "$root"    \
            --shots           "$SHOTS"   \
            --seed            "$SEED"    \
            --n_iters         "$N_ITERS" \
            --n_prompt_tokens "$N_PROMPT_TOKENS" \
            --batch_size      "$BATCH"   \
            --save_path       "$SAVE"    \
            2>&1 | tee -a "$log"
        local exit_code=${PIPESTATUS[0]}

        if [ "$exit_code" -eq 0 ]; then
            echo "[OK] $model/vpt/$dataset completed."
            return 0
        fi

        echo "[WARN] $model/vpt/$dataset attempt ${attempt}/${max_attempts} failed (exit ${exit_code})."
        attempt=$((attempt + 1))
        [ "$attempt" -le "$max_attempts" ] && { echo "[retry] backing off 60s ..."; sleep 60; }
    done

    echo "[FATAL] $model/vpt/$dataset failed after ${max_attempts} attempts, giving up (see ${log})."
    return 1
}

for MODEL in clip dino siglip2; do
    [[ "$FILTER_MODEL" != "all" && "$FILTER_MODEL" != "$MODEL" ]] && continue

    run "$MODEL" ucf101         "$DATA" "$DATA/UCF101"
    run "$MODEL" stanford_cars  "$DATA" "$DATA/stanford_cars"
    run "$MODEL" oxford_pets    "$DATA" "$DATA/oxford_pets_imagefolder"
    run "$MODEL" food101        "$DATA" "$DATA/food101"
    run "$MODEL" oxford_flowers "$DATA" "$DATA/flowers102_imagefolder"
    run "$MODEL" fgvc           "$DATA" "$DATA/fgvc_imagefolder"
    run "$MODEL" eurosat        "$DATA" "$DATA/eurosat"
    run "$MODEL" dtd            "$DATA" "$DATA/dtd"

    run "$MODEL" imagenet_v2     "$DATA" "$DATA/imagenetv2"
    run "$MODEL" imagenet_sketch "$DATA" "$DATA/sketch"
    run "$MODEL" imagenet_a      "$DATA" "$DATA/imagenet-a"
    run "$MODEL" imagenet_r      "$DATA" "$DATA/imagenet-r"

    run "$MODEL" caltech101  "$DATA/caltech-101" "$DATA/caltech-101/Caltech101"
    run "$MODEL" sun397      "$DATA"             "$HOME/.cache/huggingface/hub/datasets--1aurent--SUN397"
    run "$MODEL" medmnist    "$DATA/pathmnist_imagefolder" "$DATA/pathmnist_imagefolder"
    run "$MODEL" chexpert    "$DATA/chexpert"   "$DATA/chexpert"
done

echo ""
echo "============================================================"
echo " All VPT runs complete."
echo " Weights : $SAVE/<model>_vpt"
echo " Logs    : $LOG_DIR"
echo "============================================================"
