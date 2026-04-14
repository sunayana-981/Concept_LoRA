#!/usr/bin/env bash
# =============================================================================
# run_maple_sae_training.sh
#
# Train SAEs on MaPLe prompt-tuned CLIP activations for selected datasets
# (e.g., MedMNIST, Caltech101, EuroSAT).
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
MAPLE_MODEL_PATH="/home/sunayana/Documents/Concept_LoRA/maple_weights/eurosat/base/seed1/MultiModalPromptLearner/model.pth.tar-5"
MAPLE_CONFIG_PATH="/home/sunayana/Documents/Concept_LoRA/patchsae/configs/models/maple/vit_b16_c2_ep5_batch4_2ctx.yaml"

# --- Datasets to train ---
# Add more datasets here as needed (must exist in DATASET_INFO in tasks/utils.py)
DATASETS=("eurosat")

# --- Token budgets ---
declare -A TOTAL_TOKENS
TOTAL_TOKENS["medmnist"]=2000000
TOTAL_TOKENS["caltech101"]=100000
TOTAL_TOKENS["eurosat"]=700000

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

# --- Post-training top-activation export ---
EXPORT_TOP_ACTIVATIONS=true
EXPORT_BATCH_SIZE=16
EXPORT_TOP_NEURONS=20
EXPORT_IMAGES_PER_NEURON=10
EXPORT_MAX_TRACKED_IMAGES=25
FEATURE_DATA_SAVE_NAME="out/feature_data"
TOP_ACTIVATION_OUTPUT_ROOT="out/top_activations"

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

if [ "${EXPORT_TOP_ACTIVATIONS}" = true ] && [ ! -f "tasks/export_top_activating_artifacts.py" ]; then
    echo "[FATAL] Export script tasks/export_top_activating_artifacts.py not found"; exit 1
fi

for ds in "${DATASETS[@]}"; do
    if [ -z "${TOTAL_TOKENS[$ds]+x}" ]; then
        echo "[FATAL] Missing TOTAL_TOKENS entry for dataset: ${ds}"; exit 1
    fi
done

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

    set +e
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
    set -e

    POSTPROC_EXIT=0
    if [ ${EXIT_CODE} -eq 0 ] && [ "${EXPORT_TOP_ACTIVATIONS}" = true ]; then
        echo "[INFO] Training finished. Exporting top neuron plots/images..."
        STAMP_FILE=$(mktemp)
        touch -d "@${DATASET_START}" "${STAMP_FILE}"
        mapfile -t FINAL_CKPTS < <(find "${DATASET_CKPT_DIR}" -type f -name "final_*.pt" -newer "${STAMP_FILE}" | sort)
        rm -f "${STAMP_FILE}"

        if [ ${#FINAL_CKPTS[@]} -eq 0 ]; then
            echo "[ERROR] No final SAE checkpoints found for ${dataset} after training."
            POSTPROC_EXIT=1
        else
            for SAE_CKPT in "${FINAL_CKPTS[@]}"; do
                echo "[INFO] Exporting artifacts for checkpoint: ${SAE_CKPT}"
                set +e
                python3 tasks/export_top_activating_artifacts.py \
                    --sae_path "${SAE_CKPT}" \
                    --dataset_name "${dataset}" \
                    --vit_type maple \
                    --root_dir "${PROJECT_ROOT}" \
                    --feature_save_name "${FEATURE_DATA_SAVE_NAME}" \
                    --output_root "${TOP_ACTIVATION_OUTPUT_ROOT}" \
                    --backbone "${MODEL_NAME}" \
                    --model_path "${MAPLE_MODEL_PATH}" \
                    --config_path "${MAPLE_CONFIG_PATH}" \
                    --device "${DEVICE}" \
                    --batch_size "${EXPORT_BATCH_SIZE}" \
                    --num_top_images_per_neuron "${EXPORT_MAX_TRACKED_IMAGES}" \
                    --num_neurons_to_plot "${EXPORT_TOP_NEURONS}" \
                    --images_per_neuron_grid "${EXPORT_IMAGES_PER_NEURON}" \
                    --seed "${SEED}" \
                    2>&1 | tee -a "${RUN_LOG}"
                CURR_EXPORT_EXIT=${PIPESTATUS[0]}
                set -e

                if [ ${CURR_EXPORT_EXIT} -ne 0 ]; then
                    echo "[ERROR] Export failed for ${SAE_CKPT} (exit ${CURR_EXPORT_EXIT})"
                    POSTPROC_EXIT=${CURR_EXPORT_EXIT}
                    break
                fi
            done
        fi
    fi

    DATASET_END=$(date +%s)
    DATASET_MIN=$(( (DATASET_END - DATASET_START) / 60 ))

    if [ ${EXIT_CODE} -eq 0 ] && [ ${POSTPROC_EXIT} -eq 0 ]; then
        echo "[SUCCESS] ${dataset} (maple) done in ${DATASET_MIN}m"
        SUCCEEDED+=("${dataset}")
    else
        echo "[FAILED] ${dataset} (maple) (train exit ${EXIT_CODE}, export exit ${POSTPROC_EXIT}) — see ${RUN_LOG}"
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
