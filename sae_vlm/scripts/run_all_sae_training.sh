#!/usr/bin/env bash
# =============================================================================
# run_all_sae_training.sh
#
# Batch SAE training across multiple LoRA fine-tuned CLIP models.
# Token budgets calibrated from EuroSAT convergence at ~700K tokens.
#
# Usage:
#   chmod +x run_all_sae_training.sh
#   ./run_all_sae_training.sh 2>&1 | tee full_training_log.txt
#
# Background (SSH-safe):
#   nohup ./run_all_sae_training.sh > full_training_log.txt 2>&1 &
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION — EDIT THESE
# =============================================================================

PROJECT_ROOT="/home/sunayana/Documents/Concept_LoRA/patchsae"
LORA_WEIGHTS_ROOT="/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16"
CHECKPOINT_ROOT="/home/sunayana/Documents/Concept_LoRA/patchsae/out/checkpoints"
LOG_ROOT="/home/sunayana/Documents/Concept_LoRA/patchsae/out/logs"

# --- LoRA checkpoint paths ---
declare -A LORA_PATHS
LORA_PATHS["eurosat"]="${LORA_WEIGHTS_ROOT}/eurosat/16shots/seed1/lora_weights.pt"
LORA_PATHS["caltech101"]="${LORA_WEIGHTS_ROOT}/caltech101/16shots/seed1/lora_weights.pt"
LORA_PATHS["medmnist"]="${LORA_WEIGHTS_ROOT}/medmnist/16shots/seed1/lora_weights.pt"
LORA_PATHS["dtd"]="${LORA_WEIGHTS_ROOT}/dtd/16shots/seed42/lora_weights.pt"
LORA_PATHS["ucf101"]="${LORA_WEIGHTS_ROOT}/ucf101/16shots/seed1/lora_weights.pt"
LORA_PATHS["cub2002011"]="${LORA_WEIGHTS_ROOT}/cub2002011/16shots/seed1/lora_weights.pt"
LORA_PATHS["oxford_pets"]="${LORA_WEIGHTS_ROOT}/oxford_pets/16shots/seed1/lora_weights.pt"
LORA_PATHS["fgvc"]="${LORA_WEIGHTS_ROOT}/fgvc/16shots/seed1/lora_weights.pt"

# --- Token budgets ---
# EuroSAT converged at ~700K tokens. Using ~3x margin per dataset.
# Smaller/similar datasets: 2M tokens. Larger (MedMNIST, UCF101, CUB200): 5M tokens.
declare -A TOTAL_TOKENS
TOTAL_TOKENS["eurosat"]=2000000
TOTAL_TOKENS["caltech101"]=2000000
TOTAL_TOKENS["medmnist"]=2000000
TOTAL_TOKENS["dtd"]=2000000
TOTAL_TOKENS["ucf101"]=5000000
TOTAL_TOKENS["cub2002011"]=5000000
TOTAL_TOKENS["oxford_pets"]=2000000
TOTAL_TOKENS["fgvc"]=1000000

# --- Shared hyperparameters ---
BLOCK_LAYERS="-2"
EXPANSION_FACTOR=64
L1_COEFFICIENT=0.00008
LR=0.0004
BATCH_SIZE=16
LR_WARM_UP_STEPS=500
CLIP_DIM=768
MODEL_NAME="openai/clip-vit-base-patch16"
SEED=42
WANDB_PROJECT="lora_clip_sae"
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

check_lora_path() {
    local dataset=$1
    local path=${LORA_PATHS[$dataset]}
    if [ ! -f "$path" ]; then
        echo "[ERROR] LoRA weights not found for ${dataset}: ${path}"
        return 1
    fi
    echo "[OK] ${dataset}: ${path} ($(du -h "$path" | cut -f1))"
    return 0
}

# =============================================================================
# PRE-FLIGHT
# =============================================================================

log_header "PRE-FLIGHT CHECKS"

cd "$PROJECT_ROOT"

if [ ! -f "tasks/train_sae_lora_clip.py" ]; then
    echo "[FATAL] Training script not found"; exit 1
fi

echo "Python: $(python3 --version 2>&1)"
nvidia-smi --query-gpu=name,memory.free,temperature.gpu --format=csv,noheader 2>/dev/null || true
echo ""

# Set up imagefolder symlinks for OxfordPets and FGVC (idempotent)
echo "[INFO] Setting up imagefolder datasets (OxfordPets, FGVC)..."
python3 setup_imagefolder_datasets.py
echo ""

DATASETS_TO_TRAIN=()
for dataset in dtd ucf101 cub2002011 oxford_pets fgvc; do
    if check_lora_path "$dataset"; then
        DATASETS_TO_TRAIN+=("$dataset")
    fi
done

if [ ${#DATASETS_TO_TRAIN[@]} -eq 0 ]; then
    echo "[FATAL] No valid checkpoints found."; exit 1
fi

# Estimated times based on ~91 it/s from EuroSAT run
echo ""
echo "Training plan (estimated from your 3090 @ ~91 it/s):"
for ds in "${DATASETS_TO_TRAIN[@]}"; do
    tokens=${TOTAL_TOKENS[$ds]}
    # 2 layers, so multiply time by 2
    est_min_per_layer=$(( tokens / 91 / 60 ))
    est_total=$(( est_min_per_layer * 2 ))
    echo "  ${ds}: ${tokens} tokens → ~${est_total} min (both layers)"
done
total_est=0
for ds in "${DATASETS_TO_TRAIN[@]}"; do
    tokens=${TOTAL_TOKENS[$ds]}
    total_est=$(( total_est + tokens * 2 / 91 / 60 ))
done
echo "  TOTAL: ~${total_est} min (~$(( total_est / 60 ))h $(( total_est % 60 ))m)"
echo ""

mkdir -p "${CHECKPOINT_ROOT}" "${LOG_ROOT}"

# =============================================================================
# TRAINING
# =============================================================================

TOTAL_DATASETS=${#DATASETS_TO_TRAIN[@]}
CURRENT=0
FAILED=()
SUCCEEDED=()
OVERALL_START=$(date +%s)

for dataset in "${DATASETS_TO_TRAIN[@]}"; do
    CURRENT=$((CURRENT + 1))

    log_header "DATASET ${CURRENT}/${TOTAL_DATASETS}: ${dataset^^}"

    LORA_PATH=${LORA_PATHS[$dataset]}
    TOKENS=${TOTAL_TOKENS[$dataset]}
    DATASET_CKPT_DIR="${CHECKPOINT_ROOT}/${dataset}"
    DATASET_LOG_DIR="${LOG_ROOT}/${dataset}"
    RUN_LOG="${DATASET_LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

    mkdir -p "${DATASET_CKPT_DIR}" "${DATASET_LOG_DIR}"

    echo "  Tokens:      ${TOKENS}"
    echo "  Layers:      ${BLOCK_LAYERS}"
    echo "  Checkpoints: ${DATASET_CKPT_DIR}"
    echo "  Log:         ${RUN_LOG}"
    echo ""

    DATASET_START=$(date +%s)

    python3 tasks/train_sae_lora_clip.py \
        --model_name "${MODEL_NAME}" \
        --clip_dim ${CLIP_DIM} \
        --lora_checkpoint_path "${LORA_PATH}" \
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
        --run_name "${dataset}_sae" \
        2>&1 | tee "${RUN_LOG}"

    EXIT_CODE=${PIPESTATUS[0]}
    DATASET_END=$(date +%s)
    DATASET_MIN=$(( (DATASET_END - DATASET_START) / 60 ))

    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "[SUCCESS] ${dataset} done in ${DATASET_MIN}m"
        SUCCEEDED+=("${dataset}")
    else
        echo "[FAILED] ${dataset} (exit ${EXIT_CODE}) — see ${RUN_LOG}"
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
    echo "  ✓ ${ds} → ${CHECKPOINT_ROOT}/${ds}/"
done
for ds in "${FAILED[@]}"; do
    echo "  ✗ ${ds} → ${LOG_ROOT}/${ds}/"
done

echo ""
echo "W&B: https://wandb.ai/sunayana1233-iiit-hyderabad/${WANDB_PROJECT}"
echo ""
echo "Checkpoints:"
find "${CHECKPOINT_ROOT}" -type f \( -name "*.pt" -o -name "*.bin" \) 2>/dev/null | head -20 || echo "  (none found)"