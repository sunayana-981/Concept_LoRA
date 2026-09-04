#!/usr/bin/env bash
# =============================================================================
# run_masked_finetune_expanded.sh
#
# Masked-fine-tunes the base SAEs trained by run_train_base_sae_expanded.sh
# (3 new backbones x 2 new SAE variants) on each of a set of target datasets,
# using the *base* (non-adapter) backbone in each case -- LoRA/DoRA adapter
# loading for dino/align/siglip2 isn't wired into
# tasks/train_sae_masked_finetune.py yet (only CLIP's adapter format is
# supported there today; see that script's --backbone handling). This is
# still a real, meaningful combination: masked fine-tuning of a base-model
# SAE onto new-domain data with high-activity (ImageNet-estimated) units
# protected, just without a fine-tuned backbone in the loop -- and it's the
# one that's actually wired up and tested end-to-end right now.
#
# 3 backbones x 2 variants x 9 datasets = 54 runs.
#
# Usage:
#   bash run_masked_finetune_expanded.sh                    # all 54
#   bash run_masked_finetune_expanded.sh dino topk eurosat  # just one
#
# Idempotent: skips a (backbone, variant, dataset) triple if a final
# checkpoint already exists under
# out/checkpoints/masked_finetune_expanded/<backbone>_<variant>/<dataset>/.
# =============================================================================

set -uo pipefail

# Resolve relative to this script's own location rather than hardcoding a
# machine-specific absolute path (differs locally vs. on Turing).
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SAE_ROOT="${PROJECT_ROOT}/out/checkpoints/base_expanded"
CHECKPOINT_ROOT="${PROJECT_ROOT}/out/checkpoints/masked_finetune_expanded"
LOG_ROOT="${PROJECT_ROOT}/out/logs/masked_finetune_expanded"

cd "$PROJECT_ROOT"
mkdir -p "$CHECKPOINT_ROOT" "$LOG_ROOT"

FILTER_BACKBONE="${1:-all}"
FILTER_VARIANT="${2:-all}"
FILTER_DATASET="${3:-all}"

ALL_DATASETS=(caltech101 cityscapes cub2002011 dtd eurosat fgvc kitti officehome ucf101)

BLOCK_LAYER=-2
EXPANSION_FACTOR=64
L1_COEFFICIENT=0.00008
LR=0.0004
BATCH_SIZE=16
LR_WARM_UP_STEPS=200
# Reduced from the existing gated script's convention (2,621,440) to keep
# this initial 54-run sweep to a few hours on a single shared GPU. Re-run
# with --total_training_tokens 2621440 (or higher) for a full-quality sweep
# once these are validated. Override via TOTAL_TOKENS_OVERRIDE env var.
TOTAL_TOKENS="${TOTAL_TOKENS_OVERRIDE:-32768}"
PROTECT_FRAC=0.2
ACTIVITY_N_BATCHES=50
# See run_train_base_sae_expanded.sh's DATASET_OVERRIDE comment: same
# HF-download-vs-disk-quota tradeoff applies here.
ACTIVITY_DATASET="${ACTIVITY_DATASET_OVERRIDE:-imagenet}"
SEED=42
DEVICE=cuda

# Depth-independent (not a fixed-level glob): SparseAutoencoder.get_name()
# embeds the HF model id directly into the checkpoint filename, and for any
# backbone whose id contains a "/" (facebook/dinov2-base,
# kakaobrain/align-base, google/siglip2-base-patch16-224 -- everything
# except the CLIP path this pattern was originally written for) that slash
# becomes a real extra directory level on disk
# (.../final_sparse_autoencoder_facebook/dinov2-base_..._resid_....pt),
# which a fixed "*/final_sparse_autoencoder_*.pt" glob silently matches zero
# files against -- confirmed the hard way (this entire 54-run phase treated
# 5 genuinely-completed base SAEs as missing and bailed out on every one).
find_base_sae() {
    local root=$1
    local match
    match=$(find "$root" -type f -name "*.pt" -path "*final_sparse_autoencoder*" 2>/dev/null | sort | tail -1)
    if [ -n "$match" ]; then
        printf '%s\n' "$match"
        return 0
    fi
    return 1
}

find_final_masked_checkpoint() {
    local root=$1
    local match
    match=$(find "$root" -type f -name "*.pt" -path "*final_sparse_autoencoder*" 2>/dev/null | sort | tail -1)
    if [ -n "$match" ]; then
        printf '%s\n' "$match"
        return 0
    fi
    return 1
}

# This is a shared GPU -- another user's job can spike memory usage mid-run
# and OOM-kill us even if there was headroom at launch (observed in
# practice: a run crashed mid-step-1 and MaskedSAETrainer.fit()'s
# try/finally still wrote a "final" checkpoint from that barely-trained
# state, which find_final_masked_checkpoint would otherwise mistake for a
# genuine completion on a later invocation). Hence the headroom check
# before each attempt, the exit-code check via PIPESTATUS, and deleting the
# checkpoint dir before retrying.
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

# ---------------------------------------------------------------------------
# Scratch staging
#
# Home directories on Turing are a 50GB NFS quota, but the 54 checkpoints this
# script produces total ~13GB and the quota was already 90% full -- a job that
# writes them straight to home dies partway with "Disk quota exceeded".
# Compute nodes have a much larger scratch volume instead (14T ada-lv_scratch,
# mounted at /tmp there; /ssd_scratch on some other clusters), so train into
# scratch and copy each finished checkpoint back to home afterwards, where it's
# small enough to pull off the cluster incrementally.
#
# SCRATCH_ROOT_OVERRIDE="" disables staging entirely (train straight into
# CHECKPOINT_ROOT), which is what the local single-machine runs want.
# ---------------------------------------------------------------------------
detect_scratch_root() {
    if [ "${SCRATCH_ROOT_OVERRIDE+set}" = "set" ]; then
        # Explicitly set (possibly to "" to disable staging).
        printf '%s\n' "$SCRATCH_ROOT_OVERRIDE"
        return 0
    fi
    local candidate home_fs candidate_fs
    home_fs=$(df --output=source "$PROJECT_ROOT" 2>/dev/null | tail -1)
    for candidate in "/ssd_scratch/${USER}" "/tmp/${USER}"; do
        mkdir -p "$candidate" 2>/dev/null || continue
        candidate_fs=$(df --output=source "$candidate" 2>/dev/null | tail -1)
        # Only stage if scratch is a genuinely different filesystem -- on a
        # single-disk machine /tmp is the same volume, so staging there would
        # just double the space used while copying.
        if [ -n "$candidate_fs" ] && [ "$candidate_fs" != "$home_fs" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    printf '%s\n' ""
}

SCRATCH_ROOT="$(detect_scratch_root)"
if [ -n "$SCRATCH_ROOT" ]; then
    SCRATCH_CKPT_ROOT="${SCRATCH_ROOT}/masked_finetune_expanded"
    mkdir -p "$SCRATCH_CKPT_ROOT"
    echo "[scratch] staging checkpoints in ${SCRATCH_CKPT_ROOT} ($(df -h "$SCRATCH_ROOT" | awk 'NR==2 {print $4}') free), copying back to ${CHECKPOINT_ROOT}"
else
    echo "[scratch] no scratch volume in use; writing directly to ${CHECKPOINT_ROOT}"
fi

# Copy a finished checkpoint from scratch back to its home-dir location. The
# scratch copy is kept if the copy back fails (e.g. home quota full) so the
# result is never lost -- it just has to be collected off the node instead.
stage_back() {
    local src=$1 dest=$2
    local need_kb avail_kb
    need_kb=$(du -sk "$src" 2>/dev/null | awk '{print $1}')
    avail_kb=$(df -Pk "$(dirname "$CHECKPOINT_ROOT")" 2>/dev/null | awk 'NR==2 {print $4}')
    # Leave 1GB of headroom so a copy-back never fills the quota completely
    # and breaks a later run's logging/checkpointing.
    if [ -n "$need_kb" ] && [ -n "$avail_kb" ] && [ "$avail_kb" -lt $((need_kb + 1048576)) ]; then
        echo "[scratch] home has ${avail_kb}KB free, need ~${need_kb}KB -- leaving result on scratch: ${src}"
        echo "[scratch] (collect with: bash pull_scratch_results.sh)"
        return 1
    fi
    mkdir -p "$(dirname "$dest")"
    if cp -r "$src" "$dest" 2>/dev/null; then
        echo "[scratch] copied back -> ${dest}"
        rm -rf "$src"
        return 0
    fi
    echo "[scratch][WARN] copy back to ${dest} failed; result kept at ${src}"
    return 1
}

run() {
    local backbone=$1 variant=$2 dataset=$3

    [[ "$FILTER_BACKBONE" != "all" && "$FILTER_BACKBONE" != "$backbone" ]] && return 0
    [[ "$FILTER_VARIANT"  != "all" && "$FILTER_VARIANT"  != "$variant"  ]] && return 0
    [[ "$FILTER_DATASET"  != "all" && "$FILTER_DATASET"  != "$dataset"  ]] && return 0

    local base_sae
    base_sae=$(find_base_sae "${BASE_SAE_ROOT}/${backbone}_${variant}") || {
        echo "[FATAL] No base SAE found for ${backbone}/${variant} under ${BASE_SAE_ROOT}/${backbone}_${variant}."
        echo "        Run run_train_base_sae_expanded.sh first."
        return 1
    }

    local ckpt_dir="${CHECKPOINT_ROOT}/${backbone}_${variant}/${dataset}"
    if existing=$(find_final_masked_checkpoint "$ckpt_dir"); then
        echo "[SKIP] ${backbone}/${variant}/${dataset}: already done -> ${existing}"
        return 0
    fi

    # Train into scratch when available, then copy the finished result back to
    # ckpt_dir; the idempotency check above still keys off the home-dir copy.
    local work_dir="$ckpt_dir"
    if [ -n "$SCRATCH_ROOT" ]; then
        work_dir="${SCRATCH_CKPT_ROOT}/${backbone}_${variant}/${dataset}"
        if existing=$(find_final_masked_checkpoint "$work_dir"); then
            echo "[SKIP] ${backbone}/${variant}/${dataset}: already done on scratch -> ${existing}"
            stage_back "$work_dir" "$ckpt_dir" || true
            return 0
        fi
    fi

    local log="${LOG_ROOT}/${backbone}_${variant}_${dataset}.log"
    local variant_flag=""
    [[ "$variant" == "topk" ]] && variant_flag="--topk_sae --topk_k 32"
    [[ "$variant" == "jumprelu" ]] && variant_flag="--jumprelu_sae"

    local attempt=1 max_attempts=3
    while [ "$attempt" -le "$max_attempts" ]; do
        rm -rf "$work_dir"; mkdir -p "$work_dir"
        wait_for_gpu_headroom
        echo ""
        echo "============================================================"
        echo " MASKED FINETUNE (expanded)  |  ${backbone}/${variant}/${dataset}  |  Attempt ${attempt}/${max_attempts}"
        echo " Base SAE: ${base_sae}"
        echo "============================================================"

        python3 tasks/train_sae_masked_finetune.py \
            --sae_checkpoint_path "$base_sae" \
            --backbone "$backbone" \
            --block_layer ${BLOCK_LAYER} \
            --dataset "$dataset" \
            --protect_frac ${PROTECT_FRAC} \
            --activity_n_batches ${ACTIVITY_N_BATCHES} \
            --activity_dataset "${ACTIVITY_DATASET}" \
            --expansion_factor ${EXPANSION_FACTOR} \
            ${variant_flag} \
            --l1_coefficient ${L1_COEFFICIENT} \
            --lr ${LR} \
            --batch_size ${BATCH_SIZE} \
            --lr_warm_up_steps ${LR_WARM_UP_STEPS} \
            --total_training_tokens ${TOTAL_TOKENS} \
            --use_ghost_grads \
            --checkpoint_path "$work_dir" \
            --n_checkpoints 1 \
            --seed ${SEED} \
            --device ${DEVICE} \
            2>&1 | tee -a "$log"
        local exit_code=${PIPESTATUS[0]}

        if [ "$exit_code" -eq 0 ]; then
            echo "[OK] ${backbone}/${variant}/${dataset} completed."
            [ "$work_dir" != "$ckpt_dir" ] && { stage_back "$work_dir" "$ckpt_dir" || true; }
            return 0
        fi

        echo "[WARN] ${backbone}/${variant}/${dataset} attempt ${attempt}/${max_attempts} failed (exit ${exit_code})."
        rm -rf "$work_dir"
        attempt=$((attempt + 1))
        [ "$attempt" -le "$max_attempts" ] && { echo "[retry] backing off 60s ..."; sleep 60; }
    done

    echo "[FATAL] ${backbone}/${variant}/${dataset} failed after ${max_attempts} attempts, giving up (see ${log})."
    return 1
}

for backbone in dino align siglip2; do
    for variant in topk jumprelu; do
        for dataset in "${ALL_DATASETS[@]}"; do
            run "$backbone" "$variant" "$dataset"
        done
    done
done

echo ""
echo "============================================================"
echo " All masked-finetune (expanded) runs complete."
echo " Checkpoints: ${CHECKPOINT_ROOT}"
echo "============================================================"
