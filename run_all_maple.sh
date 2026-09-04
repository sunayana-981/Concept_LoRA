#!/usr/bin/env bash
# =============================================================================
# run_all_maple.sh
# MaPLe (Multi-modal Prompt Learning) fine-tuning on CLIP across all 16
# datasets used by run_all_unified.sh's LoRA/DoRA matrix. CLIP only -- see
# train_maple.py's docstring for why ALIGN/SigLIP2 aren't attempted.
#
# 16 datasets x 1 model = 16 runs.
#
# Usage:
#   bash run_all_maple.sh
#   bash run_all_maple.sh eurosat     # only the eurosat run
#
# Idempotent: skips a dataset if
# maple_weights/<dataset>/base/seed<seed>/<config_name>/model.pth.tar-<epoch>
# already exists. Missing dataset directories are skipped with a warning
# rather than aborting the whole sweep.
# =============================================================================

set -uo pipefail

# Resolve relative to this script's own location rather than hardcoding a
# machine-specific absolute path -- the same repo checks out under
# different absolute paths locally vs. on a remote cluster (e.g. Turing,
# where the previous hardcoded local path caused every dataset lookup to
# resolve to a nonexistent directory and silently skip all 16 runs).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA=$REPO/data
SAVE=$REPO/maple_weights
LOG_DIR=$REPO/unified_logs_maple
CONFIG_PATH=$REPO/patchsae/configs/models/maple/vit_b16_c2_ep5_batch4_2ctx.yaml
CONFIG_NAME=vit_b16_c2_ep5_batch4_2ctx

mkdir -p "$LOG_DIR"
cd "$REPO"

FILTER_DATASET="${1:-all}"

SHOTS=16
SEED=1
# The official recipe's yaml default is 5 epochs; reduced here to keep this
# initial 16-run sweep quicker on a single shared GPU. Override via
# EPOCHS_OVERRIDE env var for a full-quality sweep once these are validated.
EPOCHS="${EPOCHS_OVERRIDE:-3}"

# This is a shared GPU -- another user's job can spike memory usage mid-run
# and OOM-kill us even if there was headroom at launch. train_maple.py has
# no crash-safety-net checkpoint save, so a crash just means no output file
# -- idempotency already handles that correctly on a retry -- but a
# headroom check + backoff still meaningfully reduces how often that happens.
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
    local dataset=$1
    local root=$2
    local check_dir="${3:-$root}"

    [[ "$FILTER_DATASET" != "all" && "$FILTER_DATASET" != "$dataset" ]] && return 0

    if [[ ! -d "$check_dir" ]]; then
        echo "SKIP maple/$dataset — data directory not found: $check_dir"
        return 0
    fi

    local out="$SAVE/$dataset/base/seed${SEED}/${CONFIG_NAME}/model.pth.tar-${EPOCHS}"
    if [[ -f "$out" ]]; then
        echo "SKIP maple/$dataset — already done: $out"
        return 0
    fi

    local log="$LOG_DIR/maple_${dataset}.log"
    local attempt=1 max_attempts=3
    while [ "$attempt" -le "$max_attempts" ]; do
        wait_for_gpu_headroom
        echo ""
        echo "============================================================"
        echo " MaPLe  |  Dataset: $dataset  |  Attempt ${attempt}/${max_attempts}"
        echo "============================================================"

        python3 train_maple.py \
            --dataset     "$dataset" \
            --root_path   "$root"    \
            --shots       "$SHOTS"   \
            --seed        "$SEED"    \
            --config_path "$CONFIG_PATH" \
            --epochs      "$EPOCHS"  \
            --save_path   "$SAVE"    \
            2>&1 | tee -a "$log"
        local exit_code=${PIPESTATUS[0]}

        if [ "$exit_code" -eq 0 ]; then
            echo "[OK] maple/$dataset completed."
            return 0
        fi

        echo "[WARN] maple/$dataset attempt ${attempt}/${max_attempts} failed (exit ${exit_code})."
        attempt=$((attempt + 1))
        [ "$attempt" -le "$max_attempts" ] && { echo "[retry] backing off 60s ..."; sleep 60; }
    done

    echo "[FATAL] maple/$dataset failed after ${max_attempts} attempts, giving up (see ${log})."
    return 1
}

run ucf101         "$DATA" "$DATA/UCF101"
run stanford_cars  "$DATA" "$DATA/stanford_cars"
run oxford_pets    "$DATA" "$DATA/oxford_pets_imagefolder"
run food101        "$DATA" "$DATA/food101"
run oxford_flowers "$DATA" "$DATA/flowers102_imagefolder"
run fgvc           "$DATA" "$DATA/fgvc_imagefolder"
run eurosat        "$DATA" "$DATA/eurosat"
run dtd            "$DATA" "$DATA/dtd"

run imagenet_v2     "$DATA" "$DATA/imagenetv2"
run imagenet_sketch "$DATA" "$DATA/sketch"
run imagenet_a      "$DATA" "$DATA/imagenet-a"
run imagenet_r      "$DATA" "$DATA/imagenet-r"

run caltech101  "$DATA/caltech-101" "$DATA/caltech-101/Caltech101"
run sun397      "$DATA"             "$HOME/.cache/huggingface/hub/datasets--1aurent--SUN397"
run medmnist    "$DATA/pathmnist_imagefolder" "$DATA/pathmnist_imagefolder"
run chexpert    "$DATA/chexpert"   "$DATA/chexpert"

echo ""
echo "============================================================"
echo " All MaPLe runs complete."
echo " Weights : $SAVE"
echo " Logs    : $LOG_DIR"
echo "============================================================"
