#!/usr/bin/env bash
# =============================================================================
# run_masked_finetune_all_datasets_gated.sh
#
# Same as run_masked_finetune_all_datasets.sh, but masked-fine-tunes the
# Gated-SAE architecture base checkpoint (out/checkpoints/gsae_gated/) instead
# of the standard (ReLU+L1) G-SAE. Registers condition "masked_gated" instead
# of "masked". Requires the gated base SAE to already be trained (see
# tasks/train_sae_vit.py --gated_sae).
#
# Idempotent: a dataset is skipped if a FINAL layer -2 checkpoint already
# exists under out/checkpoints/masked_finetune_all_gated/{dataset}/.
#
# Usage:
#   chmod +x run_masked_finetune_all_datasets_gated.sh
#   ./run_masked_finetune_all_datasets_gated.sh
# =============================================================================

set -uo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT="/home/sunayana/Documents/Concept_LoRA/patchsae"
LORA_WEIGHTS_ROOT="/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16"
CHECKPOINT_ROOT="${PROJECT_ROOT}/out/checkpoints/masked_finetune_all_gated"
LOG_ROOT="${PROJECT_ROOT}/out/logs/masked_finetune_all_gated"
REGISTRY_PATH="${PROJECT_ROOT}/out/rebuttal/sae_registry.json"
GATED_BASE_ROOT="${PROJECT_ROOT}/out/checkpoints/gsae_gated"

declare -A LORA_PATHS
LORA_PATHS["caltech101"]="${LORA_WEIGHTS_ROOT}/caltech101/16shots/seed1/lora_weights.pt"
LORA_PATHS["cityscapes"]="${LORA_WEIGHTS_ROOT}/cityscapes/16shots/seed1/lora_weights.pt"
LORA_PATHS["cub2002011"]="${LORA_WEIGHTS_ROOT}/cub2002011/16shots/seed1/lora_weights.pt"
LORA_PATHS["dtd"]="${LORA_WEIGHTS_ROOT}/dtd/16shots/seed42/lora_weights.pt"
LORA_PATHS["eurosat"]="${LORA_WEIGHTS_ROOT}/eurosat/16shots/seed1/lora_weights.pt"
LORA_PATHS["fgvc"]="${LORA_WEIGHTS_ROOT}/fgvc/16shots/seed1/lora_weights.pt"
LORA_PATHS["kitti"]="${LORA_WEIGHTS_ROOT}/kitti/16shots/seed1/lora_weights.pt"
LORA_PATHS["pathmnist"]="${LORA_WEIGHTS_ROOT}/medmnist/16shots/seed1/lora_weights.pt"
LORA_PATHS["officehome"]="${LORA_WEIGHTS_ROOT}/officehome/16shots/seed1/lora_weights.pt"
LORA_PATHS["pets"]="${LORA_WEIGHTS_ROOT}/oxford_pets/16shots/seed1/lora_weights.pt"
LORA_PATHS["ucf101"]="${LORA_WEIGHTS_ROOT}/ucf101/16shots/seed1/lora_weights.pt"

declare -A TRAIN_DATASET_KEY
TRAIN_DATASET_KEY["caltech101"]="caltech101"
TRAIN_DATASET_KEY["cityscapes"]="cityscapes"
TRAIN_DATASET_KEY["cub2002011"]="cub2002011"
TRAIN_DATASET_KEY["dtd"]="dtd"
TRAIN_DATASET_KEY["eurosat"]="eurosat"
TRAIN_DATASET_KEY["fgvc"]="fgvc"
TRAIN_DATASET_KEY["kitti"]="kitti"
TRAIN_DATASET_KEY["pathmnist"]="medmnist"
TRAIN_DATASET_KEY["officehome"]="officehome"
TRAIN_DATASET_KEY["pets"]="oxford_pets"
TRAIN_DATASET_KEY["ucf101"]="ucf101"

ALL_DATASETS=(caltech101 cityscapes cub2002011 dtd eurosat fgvc kitti officehome ucf101)

BLOCK_LAYER=-2
EXPANSION_FACTOR=64
L1_COEFFICIENT=0.00008
LR=0.0004
BATCH_SIZE=16
LR_WARM_UP_STEPS=500
CLIP_DIM=768
MODEL_NAME="openai/clip-vit-base-patch16"
SEED=42
TOTAL_TOKENS=30000
PROTECT_FRAC=0.2
ACTIVITY_N_BATCHES=50
ACTIVITY_DATASET="imagenet"
WANDB_PROJECT="masked_sae_finetune_all_gated"
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

find_layer2_checkpoint() {
    local root=$1
    shopt -s nullglob
    local matches=("${root}"/*/final_sparse_autoencoder_*/*_${BLOCK_LAYER}_resid_*.pt)
    shopt -u nullglob
    if [ ${#matches[@]} -gt 0 ]; then
        printf '%s\n' "${matches[-1]}"
        return 0
    fi
    return 1
}

register_sae() {
    local dataset=$1 ckpt=$2 tokens=$3
    python3 - "$REGISTRY_PATH" "$dataset" "$ckpt" "$tokens" "$BLOCK_LAYER" <<'PYEOF'
import json, os, sys, time
registry_path, dataset, ckpt, tokens, layer = sys.argv[1:6]
os.makedirs(os.path.dirname(registry_path), exist_ok=True)
records = []
if os.path.exists(registry_path):
    with open(registry_path) as f:
        records = json.load(f)
records = [r for r in records if not (r["dataset"] == dataset and r["condition"] == "masked_gated")]
records.append({
    "dataset": dataset, "vit_type": "lora", "condition": "masked_gated",
    "checkpoint_path": ckpt, "tokens": int(tokens), "layer": int(layer),
    "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
})
with open(registry_path, "w") as f:
    json.dump(records, f, indent=2)
print(f"[REGISTRY] {dataset}/masked_gated -> {ckpt}")
PYEOF
}

# =============================================================================
# PRE-FLIGHT
# =============================================================================

log_header "PRE-FLIGHT CHECKS (GATED ARCHITECTURE)"
cd "$PROJECT_ROOT"

if [ ! -f "tasks/train_sae_masked_finetune.py" ]; then
    echo "[FATAL] tasks/train_sae_masked_finetune.py not found"; exit 1
fi

GATED_SAE_CHECKPOINT=$(find_layer2_checkpoint "$GATED_BASE_ROOT") || {
    echo "[FATAL] No gated base SAE checkpoint found under ${GATED_BASE_ROOT}."
    echo "        Run tasks/train_sae_vit.py --gated_sae first."
    exit 1
}
echo "[OK] Gated base SAE: ${GATED_SAE_CHECKPOINT}"
nvidia-smi --query-gpu=name,memory.free,temperature.gpu --format=csv,noheader 2>/dev/null || true

for dataset in "${ALL_DATASETS[@]}"; do
    path=${LORA_PATHS[$dataset]}
    if [ ! -f "$path" ]; then
        echo "[FATAL] LoRA weights not found for ${dataset}: ${path}"
        exit 1
    fi
done
echo "[OK] All 11 LoRA checkpoints present."

mkdir -p "${CHECKPOINT_ROOT}" "${LOG_ROOT}" "$(dirname "$REGISTRY_PATH")"

# =============================================================================
# IDEMPOTENCY CHECK
# =============================================================================

log_header "CHECKING FOR EXISTING LAYER ${BLOCK_LAYER} GATED-MASKED CHECKPOINTS"

DATASETS_TO_TRAIN=()
for dataset in "${ALL_DATASETS[@]}"; do
    if existing=$(find_layer2_checkpoint "${CHECKPOINT_ROOT}/${dataset}"); then
        echo "[SKIP] ${dataset}: layer ${BLOCK_LAYER} masked_gated SAE already exists -> ${existing}"
        register_sae "$dataset" "$existing" "$TOTAL_TOKENS"
    else
        echo "[TRAIN] ${dataset}: no layer ${BLOCK_LAYER} masked_gated checkpoint found"
        DATASETS_TO_TRAIN+=("$dataset")
    fi
done

if [ ${#DATASETS_TO_TRAIN[@]} -eq 0 ]; then
    log_header "NOTHING TO DO — all masked_gated SAEs already registered"
    exit 0
fi

echo ""
echo "Training plan: ${#DATASETS_TO_TRAIN[@]} dataset(s) x ${TOTAL_TOKENS} tokens each (gated architecture)"
echo "  ${DATASETS_TO_TRAIN[*]}"

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
    log_header "GATED DATASET ${CURRENT}/${TOTAL_DATASETS}: ${dataset^^}"

    LORA_PATH=${LORA_PATHS[$dataset]}
    TRAIN_KEY=${TRAIN_DATASET_KEY[$dataset]}
    DATASET_CKPT_DIR="${CHECKPOINT_ROOT}/${dataset}"
    DATASET_LOG_DIR="${LOG_ROOT}/${dataset}"
    RUN_LOG="${DATASET_LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "${DATASET_CKPT_DIR}" "${DATASET_LOG_DIR}"

    echo "  LoRA ckpt:      ${LORA_PATH}"
    echo "  train --dataset ${TRAIN_KEY}"
    echo "  Base gated SAE: ${GATED_SAE_CHECKPOINT}"
    echo "  Tokens:         ${TOTAL_TOKENS}"
    echo "  Log:            ${RUN_LOG}"
    echo ""

    DATASET_START=$(date +%s)
    DATASET_START_ISO=$(date -d "@${DATASET_START}" +"%Y-%m-%d %H:%M:%S")

    python3 tasks/train_sae_masked_finetune.py \
        --sae_checkpoint_path "${GATED_SAE_CHECKPOINT}" \
        --lora_checkpoint_path "${LORA_PATH}" \
        --model_name "${MODEL_NAME}" \
        --clip_dim ${CLIP_DIM} \
        --block_layer ${BLOCK_LAYER} \
        --dataset "${TRAIN_KEY}" \
        --protect_frac ${PROTECT_FRAC} \
        --activity_n_batches ${ACTIVITY_N_BATCHES} \
        --activity_dataset "${ACTIVITY_DATASET}" \
        --expansion_factor ${EXPANSION_FACTOR} \
        --gated_sae \
        --l1_coefficient ${L1_COEFFICIENT} \
        --lr ${LR} \
        --batch_size ${BATCH_SIZE} \
        --lr_warm_up_steps ${LR_WARM_UP_STEPS} \
        --total_training_tokens ${TOTAL_TOKENS} \
        --use_ghost_grads \
        --checkpoint_path "${DATASET_CKPT_DIR}" \
        --n_checkpoints 1 \
        --seed ${SEED} \
        --device ${DEVICE} \
        --log_to_wandb \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_log_frequency ${WANDB_LOG_FREQ} \
        --run_name "workshop_${dataset}_masked_gated" \
        2>&1 | tee "${RUN_LOG}"
    EXIT_CODE=${PIPESTATUS[0]}

    DATASET_END=$(date +%s)
    DATASET_MIN=$(( (DATASET_END - DATASET_START) / 60 ))

    if [ ${EXIT_CODE} -eq 0 ]; then
        NEW_CKPT=$(find "${DATASET_CKPT_DIR}" -newermt "${DATASET_START_ISO}" \
            -path "*/final_sparse_autoencoder_*" -name "*_${BLOCK_LAYER}_resid_*.pt" 2>/dev/null \
            | sort | tail -1)
        if [ -n "${NEW_CKPT}" ]; then
            echo "[SUCCESS] ${dataset} done in ${DATASET_MIN}m -> ${NEW_CKPT}"
            register_sae "$dataset" "$NEW_CKPT" "$TOTAL_TOKENS"
            SUCCEEDED+=("${dataset}")
        else
            echo "[WARN] ${dataset} training exited 0 but no final layer ${BLOCK_LAYER} "
            echo "       checkpoint was found under ${DATASET_CKPT_DIR} — not registered."
            FAILED+=("${dataset}")
        fi
    else
        echo "[FAILED] ${dataset} (exit ${EXIT_CODE}) — see ${RUN_LOG}"
        FAILED+=("${dataset}")
    fi

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

log_header "DONE (GATED) — ${#SUCCEEDED[@]}/${TOTAL_DATASETS} succeeded in ${OVERALL_MIN}m"
for ds in "${SUCCEEDED[@]}"; do echo "  V ${ds} -> ${CHECKPOINT_ROOT}/${ds}/"; done
for ds in "${FAILED[@]}"; do echo "  X ${ds} -> ${LOG_ROOT}/${ds}/"; done

if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi
