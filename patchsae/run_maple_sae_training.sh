#!/usr/bin/env bash
# =============================================================================
# run_maple_sae_training.sh
#
# Train SAEs on MaPLe prompt-tuned CLIP activations for MedMNIST.
#
# Uses tasks/train_sae_vit.py with --vit_type maple, which loads the
# MaPLe-adapted CLIP and hooks into transformer blocks to extract
# activations for SAE training.
#
# Usage:
#   chmod +x run_maple_sae_training.sh
#   ./run_maple_sae_training.sh 2>&1 | tee maple_training_log.txt
#
# Background (SSH-safe):
#   nohup ./run_maple_sae_training.sh > maple_training_log.txt 2>&1 &
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION — EDIT THESE
# =============================================================================

PROJECT_ROOT="/home/sunayana/Documents/Concept_LoRA/patchsae"
CHECKPOINT_ROOT="/home/sunayana/Documents/Concept_LoRA/patchsae/out/checkpoints"
LOG_ROOT="/home/sunayana/Documents/Concept_LoRA/patchsae/out/logs"

# --- MaPLe checkpoint and config ---
MAPLE_MODEL_PATH="/home/sunayana/Documents/model.pth.tar-5"
MAPLE_CONFIG_PATH="/home/sunayana/Documents/Concept_LoRA/patchsae/configs/models/maple/vit_b16_c2_ep5_batch4_2ctx.yaml"

# --- Datasets to train ---
# Add more datasets here as needed (must exist in DATASET_INFO in tasks/utils.py)
DATASETS=("medmnist")

# --- Token budgets ---
declare -A TOTAL_TOKENS
TOTAL_TOKENS["medmnist"]=2000000

# --- Shared hyperparameters ---
BLOCK_LAYERS="-2"           # which transformer block(s); use space-separated for multiple e.g. "-3 -2"
EXPANSION_FACTOR=64
L1_COEFFICIENT=0.00008
LR=0.0004
BATCH_SIZE=16
LR_WARM_UP_STEPS=500
CLIP_DIM=768
MODEL_NAME="openai/clip-vit-base-patch16"
SEED=42
WANDB_PROJECT="maple_clip_sae"
WANDB_LOG_FREQ=20
DEVICE="cuda"
export CUDA_VISIBLE_DEVICES=0

# =============================================================================
# HELPERS
# =============================================================================

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

log_header() {
    echo ""
    echo "###################################################################"
    echo "# $1"
    echo "# $(timestamp)"
    echo "###################################################################"
    echo ""
}

# =============================================================================
# PRE-FLIGHT
# =============================================================================

log_header "PRE-FLIGHT CHECKS"

cd "$PROJECT_ROOT"

if [ ! -f "tasks/train_sae_vit.py" ]; then
    echo "[FATAL] Training script tasks/train_sae_vit.py not found"; exit 1
fi

# Validate MaPLe checkpoint
if [ ! -f "${MAPLE_MODEL_PATH}" ]; then
    echo "[FATAL] MaPLe model checkpoint not found: ${MAPLE_MODEL_PATH}"; exit 1
fi
echo "[OK] MaPLe checkpoint: ${MAPLE_MODEL_PATH} ($(du -h "${MAPLE_MODEL_PATH}" | cut -f1))"

if [ ! -f "${MAPLE_CONFIG_PATH}" ]; then
    echo "[FATAL] MaPLe config not found: ${MAPLE_CONFIG_PATH}"; exit 1
fi
echo "[OK] MaPLe config:     ${MAPLE_CONFIG_PATH}"

echo "Python: $(python3 --version 2>&1)"
nvidia-smi --query-gpu=name,memory.free,temperature.gpu --format=csv,noheader 2>/dev/null || true
echo ""

# Estimated times based on ~91 it/s
echo "Training plan (estimated from ~91 it/s):"
for ds in "${DATASETS[@]}"; do
    tokens=${TOTAL_TOKENS[$ds]}
    est_min=$(( tokens / 91 / 60 ))
    echo "  ${ds}: ${tokens} tokens → ~${est_min} min per layer"
done
echo ""

mkdir -p "${CHECKPOINT_ROOT}" "${LOG_ROOT}"

# =============================================================================
# TRAINING
# =============================================================================

TOTAL_DATASETS=${#DATASETS[@]}
CURRENT=0
FAILED=()
SUCCEEDED=()
OVERALL_START=$(date +%s)

for dataset in "${DATASETS[@]}"; do
    CURRENT=$((CURRENT + 1))

    log_header "DATASET ${CURRENT}/${TOTAL_DATASETS}: ${dataset^^} (MaPLe)"

    TOKENS=${TOTAL_TOKENS[$dataset]}
    DATASET_CKPT_DIR="${CHECKPOINT_ROOT}/${dataset}_maple"
    DATASET_LOG_DIR="${LOG_ROOT}/${dataset}_maple"
    RUN_LOG="${DATASET_LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

    mkdir -p "${DATASET_CKPT_DIR}" "${DATASET_LOG_DIR}"

    echo "  VIT type:    maple"
    echo "  Tokens:      ${TOKENS}"
    echo "  Layers:      ${BLOCK_LAYERS}"
    echo "  Checkpoints: ${DATASET_CKPT_DIR}"
    echo "  Log:         ${RUN_LOG}"
    echo ""

    DATASET_START=$(date +%s)

    python3 tasks/train_sae_vit.py \
        --model_name "${MODEL_NAME}" \
        --clip_dim ${CLIP_DIM} \
        --block_layers ${BLOCK_LAYERS} \
        --dataset "${dataset}" \
        --expansion_factor ${EXPANSION_FACTOR} \
        --l1_coefficient ${L1_COEFFICIENT} \
        --lr ${LR} \
        --batch_size ${BATCH_SIZE} \
        --lr_warm_up_steps ${LR_WARM_UP_STEPS} \
        --total_training_tokens ${TOKENS} \
        --use_ghost_grads \
        --checkpoint_path "${DATASET_CKPT_DIR}" \
        --n_checkpoints 3 \
        --seed ${SEED} \
        --device ${DEVICE} \
        --log_to_wandb \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_log_frequency ${WANDB_LOG_FREQ} \
        --run_name "${dataset}_maple_sae" \
        --vit_type maple \
        --model_path "${MAPLE_MODEL_PATH}" \
        --config_path "${MAPLE_CONFIG_PATH}" \
        2>&1 | tee "${RUN_LOG}"

    EXIT_CODE=${PIPESTATUS[0]}
    DATASET_END=$(date +%s)
    DATASET_MIN=$(( (DATASET_END - DATASET_START) / 60 ))

    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "[SUCCESS] ${dataset} (maple) done in ${DATASET_MIN}m"
        SUCCEEDED+=("${dataset}")
    else
        echo "[FAILED] ${dataset} (maple) (exit ${EXIT_CODE}) — see ${RUN_LOG}"
        FAILED+=("${dataset}")
    fi

    # Clear GPU memory between datasets
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

    if [ ${CURRENT} -lt ${TOTAL_DATASETS} ]; then
        echo "[INFO] Pausing 15s before next dataset..."
        sleep 15
    fi
done

# =============================================================================
# SUMMARY
# =============================================================================

OVERALL_END=$(date +%s)
OVERALL_MIN=$(( (OVERALL_END - OVERALL_START) / 60 ))

log_header "DONE — ${#SUCCEEDED[@]}/${TOTAL_DATASETS} succeeded in ${OVERALL_MIN}m"

for ds in "${SUCCEEDED[@]}"; do
    echo "  ✓ ${ds} → ${CHECKPOINT_ROOT}/${ds}_maple/"
done
for ds in "${FAILED[@]}"; do
    echo "  ✗ ${ds} → ${LOG_ROOT}/${ds}_maple/"
done

echo ""
echo "W&B: https://wandb.ai/sunayana1233-iiit-hyderabad/${WANDB_PROJECT}"
echo ""
echo "Checkpoints:"
find "${CHECKPOINT_ROOT}" -type f \( -name "*.pt" -o -name "*.bin" \) 2>/dev/null | head -20 || echo "  (none found)"
