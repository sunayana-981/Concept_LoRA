#!/usr/bin/env bash
# =============================================================================
# run_base_vs_clipood_quick.sh
#
# Trains two SAEs on the pacs_no_photo split back-to-back:
#   1. Base CLIP SAE  (clip_vit_b16, no adapter)
#   2. CLIPood SAE    (clipood_vit_b16, photo held-out checkpoint)
#
# Tuned for ~12 min per run (~25 min total) at 170 tokens/sec.
#
# Usage:
#   ./scripts/run_base_vs_clipood_quick.sh 2>&1 | tee out/logs/base_vs_clipood.log
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CLIPOOD_CKPT="${PROJECT_ROOT}/out/clipood_pacs/clipood_pacs_held_photo.pt"
BASE_OUT="${PROJECT_ROOT}/out/checkpoints/base_pacs_no_photo_quick"
CLIPOOD_OUT="${PROJECT_ROOT}/out/checkpoints/clipood_pacs_no_photo_quick"

# 8000 steps × batch 16 = 128 K tokens ≈ 12.5 min at 170 tok/s
TOKENS=128000
BATCH=16
WARMUP=100
LAYER="-2"

ts() { date "+%Y-%m-%d %H:%M:%S"; }

if [ ! -f "${CLIPOOD_CKPT}" ]; then
    echo "[FATAL] CLIPood checkpoint not found: ${CLIPOOD_CKPT}"
    echo "  Run tasks/train_clipood_pacs.py --held_out photo first."
    exit 1
fi

# =============================================================================
# 1. Base CLIP SAE
# =============================================================================
echo ""
echo "### $(ts) | RUN 1/2: Base CLIP SAE on pacs_no_photo"
echo ""

python3 train_sae.py \
    --model             clip_vit_b16 \
    --dataset           pacs_no_photo \
    --layers            ${LAYER} \
    --total_training_tokens ${TOKENS} \
    --batch_size        ${BATCH} \
    --lr_warm_up_steps  ${WARMUP} \
    --expansion_factor  64 \
    --l1_coefficient    0.00008 \
    --lr                0.0004 \
    --checkpoint_path   "${BASE_OUT}" \
    --device            cuda

echo ""
echo "### $(ts) | Base CLIP SAE done → ${BASE_OUT}"

# =============================================================================
# 2. CLIPood SAE
# =============================================================================
echo ""
echo "### $(ts) | RUN 2/2: CLIPood SAE on pacs_no_photo (held_out=photo)"
echo ""

python3 train_sae.py \
    --model               clipood_vit_b16 \
    --dataset             pacs_no_photo \
    --layers              ${LAYER} \
    --clipood_checkpoint  "${CLIPOOD_CKPT}" \
    --total_training_tokens ${TOKENS} \
    --batch_size          ${BATCH} \
    --lr_warm_up_steps    ${WARMUP} \
    --expansion_factor    64 \
    --l1_coefficient      0.00008 \
    --lr                  0.0004 \
    --checkpoint_path     "${CLIPOOD_OUT}" \
    --device              cuda

echo ""
echo "### $(ts) | CLIPood SAE done → ${CLIPOOD_OUT}"
echo ""
echo "========================================"
echo "Both SAEs ready. Checkpoint locations:"
echo "  Base   : ${BASE_OUT}"
echo "  CLIPood: ${CLIPOOD_OUT}"
echo ""
echo "Quick eval (zero-shot acc on PACS domains):"
echo "  python eval_sae.py --sae_path ${BASE_OUT}/final_*.pt --model clip_vit_b16 --datasets pacs_no_photo --tasks feature_stats"
echo "========================================"
