#!/usr/bin/env bash
# =============================================================================
# run_train_base_sae_expanded.sh
#
# Trains base (pre-masked-fine-tune) SAEs on ImageNet activations for the
# combinations of new backbone (dino, align, siglip2) x new SAE architecture
# (topk, jumprelu) added to the masked-SAE-finetune pipeline. These are the
# prerequisite checkpoints run_masked_finetune_expanded.sh masked-finetunes
# per-dataset -- same two-step structure as the existing
# tasks/train_sae_vit.py --gated_sae -> run_masked_finetune_all_datasets_gated.sh
# pattern, just extended along both new axes at once.
#
# 3 backbones x 2 variants = 6 base SAEs.
#
# Usage:
#   bash run_train_base_sae_expanded.sh                # all 6
#   bash run_train_base_sae_expanded.sh dino topk       # just one
#
# Idempotent: skips a (backbone, variant) pair if a final checkpoint already
# exists under out/checkpoints/base_expanded/<backbone>_<variant>/.
# =============================================================================

set -uo pipefail

# Resolve relative to this script's own location rather than hardcoding a
# machine-specific absolute path (differs locally vs. on Turing).
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_ROOT="${PROJECT_ROOT}/out/checkpoints/base_expanded"
LOG_ROOT="${PROJECT_ROOT}/out/logs/base_expanded"

cd "$PROJECT_ROOT"
mkdir -p "$CHECKPOINT_ROOT" "$LOG_ROOT"

FILTER_BACKBONE="${1:-all}"   # dino | align | siglip2 | all
FILTER_VARIANT="${2:-all}"    # topk | jumprelu | all

BLOCK_LAYER=-2
EXPANSION_FACTOR=64
L1_COEFFICIENT=0.00008
LR=0.0004
BATCH_SIZE=16
LR_WARM_UP_STEPS=500
# Reduced from tasks/train_sae_vit.py's own default (2,621,440) to keep this
# initial 6-run sweep to a few hours on a single shared GPU. Re-run with
# --total_training_tokens 2621440 (or higher) for publication-quality
# dictionaries once these are validated.
TOTAL_TOKENS="${TOTAL_TOKENS_OVERRIDE:-131072}"
# "imagenet" streams evanarlian/imagenet_1k_resized_256 from the HF cache
# (already present locally, ~24GB); "imagenet_local" reads the plain
# ImageFolder at data/imagenet/train that's already on disk everywhere this
# repo is checked out, with no download at all -- use that on hosts with
# limited disk quota (e.g. Turing, where the HF download filled the quota
# mid-job) via DATASET_OVERRIDE=imagenet_local.
DATASET="${DATASET_OVERRIDE:-imagenet}"
SEED=42
DEVICE=cuda

find_final_checkpoint() {
    # Depth-independent (not a fixed-level glob): SparseAutoencoder.get_name()
    # embeds the HF model id directly into the checkpoint filename, and for
    # any backbone whose id contains a "/" (facebook/dinov2-base,
    # kakaobrain/align-base, google/siglip2-base-patch16-224 -- everything
    # except the CLIP path this pattern was originally written for) that
    # slash becomes a real extra directory level on disk
    # (.../final_sparse_autoencoder_facebook/dinov2-base_..._resid_....pt),
    # which a fixed "*/final_sparse_autoencoder_*.pt" glob silently matches
    # zero files against -- confirmed the hard way (a whole sweep phase
    # treated 5 genuinely-completed base SAEs as missing).
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
# practice). SAETrainer.fit()'s try/finally still writes a "final"
# checkpoint from whatever partial state existed when the exception hit, so
# a crashed run can otherwise be mistaken for a genuine completion by
# find_final_checkpoint on the next invocation -- hence the headroom check
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

# Train into node scratch (large) rather than the home NFS quota (50GB, and
# already near-full on Turing), copying each finished checkpoint back after.
# See run_masked_finetune_expanded.sh for the full rationale.
# SCRATCH_ROOT_OVERRIDE="" disables staging (what local runs want).
detect_scratch_root() {
    if [ "${SCRATCH_ROOT_OVERRIDE+set}" = "set" ]; then
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
    SCRATCH_CKPT_ROOT="${SCRATCH_ROOT}/base_expanded"
    mkdir -p "$SCRATCH_CKPT_ROOT"
    echo "[scratch] staging checkpoints in ${SCRATCH_CKPT_ROOT} ($(df -h "$SCRATCH_ROOT" | awk 'NR==2 {print $4}') free), copying back to ${CHECKPOINT_ROOT}"
else
    echo "[scratch] no scratch volume in use; writing directly to ${CHECKPOINT_ROOT}"
fi

stage_back() {
    local src=$1 dest=$2
    local need_kb avail_kb
    need_kb=$(du -sk "$src" 2>/dev/null | awk '{print $1}')
    avail_kb=$(df -Pk "$(dirname "$CHECKPOINT_ROOT")" 2>/dev/null | awk 'NR==2 {print $4}')
    # Leave 1GB of headroom so a copy-back never fills the quota completely.
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
    local backbone=$1 variant=$2
    [[ "$FILTER_BACKBONE" != "all" && "$FILTER_BACKBONE" != "$backbone" ]] && return 0
    [[ "$FILTER_VARIANT"  != "all" && "$FILTER_VARIANT"  != "$variant"  ]] && return 0

    local ckpt_dir="${CHECKPOINT_ROOT}/${backbone}_${variant}"
    if existing=$(find_final_checkpoint "$ckpt_dir"); then
        echo "[SKIP] ${backbone}/${variant}: base SAE already exists -> ${existing}"
        return 0
    fi

    # Train into scratch when available, copy back to ckpt_dir afterwards.
    local work_dir="$ckpt_dir"
    if [ -n "$SCRATCH_ROOT" ]; then
        work_dir="${SCRATCH_CKPT_ROOT}/${backbone}_${variant}"
        if existing=$(find_final_checkpoint "$work_dir"); then
            echo "[SKIP] ${backbone}/${variant}: already done on scratch -> ${existing}"
            stage_back "$work_dir" "$ckpt_dir" || true
            return 0
        fi
    fi

    local log="${LOG_ROOT}/${backbone}_${variant}.log"
    local variant_flag=""
    [[ "$variant" == "topk" ]] && variant_flag="--topk_sae --topk_k 32"
    [[ "$variant" == "jumprelu" ]] && variant_flag="--jumprelu_sae"

    local attempt=1 max_attempts=3
    while [ "$attempt" -le "$max_attempts" ]; do
        rm -rf "$work_dir"; mkdir -p "$work_dir"
        wait_for_gpu_headroom
        echo ""
        echo "============================================================"
        echo " BASE SAE  |  Backbone: ${backbone}  |  Variant: ${variant}  |  Attempt ${attempt}/${max_attempts}"
        echo "============================================================"

        python3 tasks/train_sae_vit.py \
            --backbone "$backbone" \
            --dataset "$DATASET" \
            --block_layers ${BLOCK_LAYER} \
            --expansion_factor ${EXPANSION_FACTOR} \
            --l1_coefficient ${L1_COEFFICIENT} \
            ${variant_flag} \
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
            echo "[OK] ${backbone}/${variant} completed."
            [ "$work_dir" != "$ckpt_dir" ] && { stage_back "$work_dir" "$ckpt_dir" || true; }
            return 0
        fi

        echo "[WARN] ${backbone}/${variant} attempt ${attempt}/${max_attempts} failed (exit ${exit_code})."
        rm -rf "$ckpt_dir"
        attempt=$((attempt + 1))
        [ "$attempt" -le "$max_attempts" ] && { echo "[retry] backing off 60s ..."; sleep 60; }
    done

    echo "[FATAL] ${backbone}/${variant} failed after ${max_attempts} attempts, giving up (see ${log})."
    return 1
}

for backbone in dino align siglip2; do
    for variant in topk jumprelu; do
        run "$backbone" "$variant"
    done
done

echo ""
echo "============================================================"
echo " All base-SAE (expanded) runs complete."
echo " Checkpoints: ${CHECKPOINT_ROOT}"
echo "============================================================"
