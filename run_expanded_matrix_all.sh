#!/usr/bin/env bash
# =============================================================================
# run_expanded_matrix_all.sh
#
# Master orchestrator for the expanded matrix: runs each new-architecture /
# new-SAE-variant / new-technique sweep script in sequence (single shared
# GPU -- these are intentionally NOT run in parallel with each other) and
# writes one combined log. Every underlying script is independently
# idempotent (skip-if-checkpoint-exists), so this can be safely re-run or
# resumed after an interruption.
#
# Order (cheapest/lowest-risk first, per the implementation plan):
#   1. Base SAEs for dino/align/siglip2 x topk/jumprelu           (6 runs)
#   2. Masked-finetune those base SAEs across 9 datasets           (54 runs)
#   3. Visual Prompt Tuning on clip/dino/siglip2 x 16 datasets     (48 runs)
#   4. MaPLe fine-tuning on clip x 16 datasets                     (16 runs)
#
# All four scripts currently use reduced (not full-paper-scale) token/epoch
# budgets to keep total wall-clock to roughly a working day on a single
# shared RTX 3090 -- see each script's own comments for the *_OVERRIDE env
# var that restores full-scale settings for a later, longer follow-up run.
# =============================================================================

set -uo pipefail

REPO=/home/sunayana/Documents/Concept_LoRA
LOG=$REPO/results/expanded_matrix_run.log

mkdir -p "$REPO/results"
cd "$REPO"

section() {
    echo ""
    echo "################################################################"
    echo "# $1"
    echo "# $(date "+%Y-%m-%d %H:%M:%S")"
    echo "################################################################"
}

# This is a shared GPU -- other users' jobs may already be running. Wait for
# at least MIN_FREE_MIB free before each phase rather than launching into
# contention (our own runs use only a few GB at a time per the smoke tests,
# so this is a courtesy margin, not a hard requirement).
MIN_FREE_MIB=4000
wait_for_gpu_headroom() {
    while true; do
        local used total free
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        free=$((total - used))
        if [ "$free" -ge "$MIN_FREE_MIB" ]; then
            echo "[gpu-check] ${free} MiB free (of ${total}) -- proceeding."
            return 0
        fi
        echo "[gpu-check] only ${free} MiB free (of ${total}), waiting for ${MIN_FREE_MIB}+ ..."
        sleep 60
    done
}

{
    section "1/4: BASE SAEs (dino/align/siglip2 x topk/jumprelu)"
    wait_for_gpu_headroom
    (cd patchsae && bash run_train_base_sae_expanded.sh)

    section "2/4: MASKED FINETUNE, EXPANDED (54 runs)"
    wait_for_gpu_headroom
    (cd patchsae && bash run_masked_finetune_expanded.sh)

    section "3/4: VISUAL PROMPT TUNING (48 runs)"
    wait_for_gpu_headroom
    bash run_all_vpt.sh

    section "4/4: MAPLE (16 runs)"
    wait_for_gpu_headroom
    bash run_all_maple.sh

    section "ALL EXPANDED-MATRIX SWEEPS COMPLETE"
} 2>&1 | tee "$LOG"
