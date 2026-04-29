#!/usr/bin/env bash
# =============================================================================
# run_clipood_pacs_sae.sh
#
# End-to-end pipeline:
#   1. Download / verify PACS data
#   2. Set up leave-one-out ImageFolder splits (symlinks)
#   3. Fine-tune CLIPood on each of the 4 splits (~30 min/split on 3090)
#   4. Train SAEs on each CLIPood model's activations (~20 min/split)
#
# Usage:
#   PACS_ROOT=/path/to/PACS ./scripts/run_clipood_pacs_sae.sh
#
# SSH-safe background run:
#   PACS_ROOT=/path/to/PACS nohup ./scripts/run_clipood_pacs_sae.sh \
#       > out/logs/clipood_pacs_full.log 2>&1 &
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION — edit these or set as env vars before running
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACS_ROOT="${PACS_ROOT:-/path/to/PACS}"       # SET THIS: dir with art_painting/ cartoon/ photo/ sketch/
CLIPOOD_OUT="${PROJECT_ROOT}/out/clipood_pacs"
IMAGEFOLDER_ROOT="${HOME}/Documents/Concept_LoRA/data/pacs_imagefolder"
CHECKPOINT_ROOT="${PROJECT_ROOT}/out/checkpoints"
LOG_ROOT="${PROJECT_ROOT}/out/logs/clipood_pacs"

# CLIPood fine-tuning hyperparams (CLIPood paper defaults for PACS)
CLIPOOD_STEPS=4000
CLIPOOD_LR=5e-6
CLIPOOD_BATCH=32

# SAE hyperparams (same as existing LoRA runs for fair comparison)
SAE_LAYER="-2"
SAE_TOKENS=2000000
SAE_EXPANSION=64
SAE_L1=0.00008
SAE_LR=0.0004
SAE_BATCH=16
SAE_WARMUP=500

DEVICE="cuda"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DOMAINS=("photo" "art_painting" "cartoon" "sketch")
SPLIT_NAMES=("no_photo" "no_art_painting" "no_cartoon" "no_sketch")

# =============================================================================
# HELPERS
# =============================================================================

ts()  { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo ""; echo "### $(ts) | $*"; echo ""; }

# =============================================================================
# PRE-FLIGHT
# =============================================================================

log "PRE-FLIGHT CHECKS"
cd "$PROJECT_ROOT"

echo "Python: $(python3 --version 2>&1)"
nvidia-smi --query-gpu=name,memory.free,temperature.gpu --format=csv,noheader 2>/dev/null || true

if [ "$PACS_ROOT" = "/path/to/PACS" ]; then
    echo "[FATAL] Set PACS_ROOT before running. Example:"
    echo "  PACS_ROOT=/home/sunayana/data/PACS ./scripts/run_clipood_pacs_sae.sh"
    exit 1
fi

for dom in art_painting cartoon photo sketch; do
    if [ ! -d "${PACS_ROOT}/${dom}" ]; then
        echo "[FATAL] Missing PACS domain: ${PACS_ROOT}/${dom}"
        exit 1
    fi
done
echo "[OK] PACS root: ${PACS_ROOT}"

mkdir -p "${CLIPOOD_OUT}" "${CHECKPOINT_ROOT}" "${LOG_ROOT}"

# =============================================================================
# STEP 1 — SET UP IMAGEFOLDER SPLITS
# =============================================================================

log "STEP 1/3: Setting up PACS ImageFolder splits"

python3 tasks/setup_pacs_imagefolder.py \
    --pacs_root  "${PACS_ROOT}" \
    --output_root "${IMAGEFOLDER_ROOT}"

echo "[OK] ImageFolder splits ready at: ${IMAGEFOLDER_ROOT}"

# =============================================================================
# STEP 2 — CLIPOOD FINE-TUNING (4 splits)
# =============================================================================

log "STEP 2/3: CLIPood fine-tuning on all 4 PACS leave-one-out splits"

CLIPOOD_FAILED=()

for held in "${DOMAINS[@]}"; do
    ckpt="${CLIPOOD_OUT}/clipood_pacs_held_${held}.pt"

    if [ -f "$ckpt" ]; then
        echo "[SKIP] CLIPood checkpoint already exists: ${ckpt}"
        continue
    fi

    log "  CLIPood: held_out=${held}"
    run_log="${LOG_ROOT}/clipood_held_${held}.log"

    python3 tasks/train_clipood_pacs.py \
        --held_out    "${held}" \
        --pacs_root   "${PACS_ROOT}" \
        --output_dir  "${CLIPOOD_OUT}" \
        --steps       ${CLIPOOD_STEPS} \
        --lr          ${CLIPOOD_LR} \
        --batch_size  ${CLIPOOD_BATCH} \
        --device      "${DEVICE}" \
        2>&1 | tee "${run_log}"

    exit_code=${PIPESTATUS[0]}
    if [ ${exit_code} -ne 0 ]; then
        echo "[FAILED] CLIPood held_out=${held} (exit ${exit_code})"
        CLIPOOD_FAILED+=("${held}")
    else
        echo "[OK] Saved: ${ckpt}"
    fi

    # Brief pause to clear GPU memory
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 5
done

if [ ${#CLIPOOD_FAILED[@]} -gt 0 ]; then
    echo "[WARN] CLIPood failed for: ${CLIPOOD_FAILED[*]} — SAE training will skip those splits"
fi

# =============================================================================
# STEP 3 — SAE TRAINING on each CLIPood model
# =============================================================================

log "STEP 3/3: SAE training on CLIPood activations (layer ${SAE_LAYER})"

SAE_FAILED=()
SAE_DONE=()

for i in "${!DOMAINS[@]}"; do
    held="${DOMAINS[$i]}"
    split="${SPLIT_NAMES[$i]}"
    clipood_ckpt="${CLIPOOD_OUT}/clipood_pacs_held_${held}.pt"
    sae_ckpt_dir="${CHECKPOINT_ROOT}/clipood_pacs_${split}"
    dataset_name="pacs_${split}"

    if [ ! -f "${clipood_ckpt}" ]; then
        echo "[SKIP] No CLIPood checkpoint for held_out=${held} — skipping SAE"
        continue
    fi

    log "  SAE: held_out=${held} | dataset=${dataset_name}"
    run_log="${LOG_ROOT}/sae_${split}.log"

    python3 train_sae.py \
        --model               clipood_vit_b16 \
        --dataset             "${dataset_name}" \
        --layers              ${SAE_LAYER} \
        --clipood_checkpoint  "${clipood_ckpt}" \
        --total_training_tokens ${SAE_TOKENS} \
        --expansion_factor    ${SAE_EXPANSION} \
        --l1_coefficient      ${SAE_L1} \
        --lr                  ${SAE_LR} \
        --batch_size          ${SAE_BATCH} \
        --lr_warm_up_steps    ${SAE_WARMUP} \
        --checkpoint_path     "${sae_ckpt_dir}" \
        --device              "${DEVICE}" \
        2>&1 | tee "${run_log}"

    exit_code=${PIPESTATUS[0]}
    if [ ${exit_code} -ne 0 ]; then
        echo "[FAILED] SAE split=${split} (exit ${exit_code})"
        SAE_FAILED+=("${split}")
    else
        echo "[OK] SAE checkpoint: ${sae_ckpt_dir}"
        SAE_DONE+=("${split}")
    fi

    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 5
done

# =============================================================================
# SUMMARY
# =============================================================================

log "DONE"
echo "CLIPood checkpoints:"
ls -lh "${CLIPOOD_OUT}"/*.pt 2>/dev/null || echo "  (none)"
echo ""
echo "SAE checkpoints:"
for split in "${SPLIT_NAMES[@]}"; do
    dir="${CHECKPOINT_ROOT}/clipood_pacs_${split}"
    if [ -d "$dir" ]; then
        n=$(find "$dir" -name "*.pt" | wc -l)
        echo "  ${split}: ${n} .pt files → ${dir}"
    fi
done
echo ""
if [ ${#SAE_FAILED[@]} -gt 0 ]; then
    echo "[WARN] Failed splits: ${SAE_FAILED[*]}"
fi
echo "Logs: ${LOG_ROOT}/"
