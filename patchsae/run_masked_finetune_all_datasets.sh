#!/usr/bin/env bash
# =============================================================================
# run_masked_finetune_all_datasets.sh
#
# Masked fine-tune of the pre-trained ImageNet G-SAE (data/sae_weight/base/out.pt)
# on LoRA-CLIP activations, for all 11 workshop-paper datasets. Extends
# run_rebuttal_sae_training.sh's pattern (same fixed hyperparameters: layer -2,
# expansion factor 64, L1 8e-5, seed 42) but uses tasks/train_sae_masked_finetune.py
# instead of tasks/train_sae_lora_clip.py, so units estimated as active on
# ImageNet (--activity_dataset imagenet) are frozen and only free capacity
# adapts to each new dataset.
#
# Idempotent: a dataset is skipped if a FINAL layer -2 checkpoint already
# exists under out/checkpoints/masked_finetune_all/{dataset}/.
#
# Registers every (dataset, "masked") checkpoint in out/rebuttal/sae_registry.json,
# which tasks/registry_to_sae_paths.py turns into the --sae_paths JSON that
# tasks/eval_matrix.py consumes.
#
# Usage:
#   chmod +x run_masked_finetune_all_datasets.sh
#   nohup ./run_masked_finetune_all_datasets.sh > out/logs/masked_finetune_all/driver.log 2>&1 &
# =============================================================================

set -uo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT="/home/sunayana/Documents/Concept_LoRA/patchsae"
LORA_WEIGHTS_ROOT="/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16"
CHECKPOINT_ROOT="${PROJECT_ROOT}/out/checkpoints/masked_finetune_all"
LOG_ROOT="${PROJECT_ROOT}/out/logs/masked_finetune_all"
REGISTRY_PATH="${PROJECT_ROOT}/out/rebuttal/sae_registry.json"
SAE_CHECKPOINT="${PROJECT_ROOT}/data/sae_weight/base/out.pt"

# --- rebuttal-facing dataset name -> LoRA checkpoint path ---
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

# --- rebuttal-facing dataset name -> tasks/utils.py DATASET_INFO key ---
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

ALL_DATASETS=(caltech101 cityscapes cub2002011 dtd eurosat fgvc kitti pathmnist officehome pets ucf101)

# --- Fixed hyperparameters (rebuttal convention, never vary across conditions) ---
BLOCK_LAYER=-2
EXPANSION_FACTOR=64
L1_COEFFICIENT=0.00008
LR=0.0004
BATCH_SIZE=16
LR_WARM_UP_STEPS=500
CLIP_DIM=768
MODEL_NAME="openai/clip-vit-base-patch16"
SEED=42
TOTAL_TOKENS=100000
PROTECT_FRAC=0.2
ACTIVITY_N_BATCHES=50
ACTIVITY_DATASET="imagenet"
WANDB_PROJECT="masked_sae_finetune_all"
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
    local dataset=$1
    shopt -s nullglob
    local matches=("${CHECKPOINT_ROOT}/${dataset}"/*/final_sparse_autoencoder_*/*_${BLOCK_LAYER}_resid_*.pt)
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
records = [r for r in records if not (r["dataset"] == dataset and r["condition"] == "masked")]
records.append({
    "dataset": dataset, "vit_type": "lora", "condition": "masked",
    "checkpoint_path": ckpt, "tokens": int(tokens), "layer": int(layer),
    "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
})
with open(registry_path, "w") as f:
    json.dump(records, f, indent=2)
print(f"[REGISTRY] {dataset}/masked -> {ckpt}")
PYEOF
}

# =============================================================================
# PRE-FLIGHT
# =============================================================================

log_header "PRE-FLIGHT CHECKS"
cd "$PROJECT_ROOT"

if [ ! -f "tasks/train_sae_masked_finetune.py" ]; then
    echo "[FATAL] tasks/train_sae_masked_finetune.py not found"; exit 1
fi
if [ ! -f "$SAE_CHECKPOINT" ]; then
    echo "[FATAL] G-SAE checkpoint not found: $SAE_CHECKPOINT"; exit 1
fi
echo "Python: $(python3 --version 2>&1)"
nvidia-smi --query-gpu=name,memory.free,temperature.gpu --format=csv,noheader 2>/dev/null || true

DATASETS_TO_TRAIN=()
for dataset in "${ALL_DATASETS[@]}"; do
    path=${LORA_PATHS[$dataset]}
    if [ ! -f "$path" ]; then
        echo "[FATAL] LoRA weights not found for ${dataset}: ${path}"
        exit 1
    fi
    echo "[OK] ${dataset}: ${path} ($(du -h "$path" | cut -f1))"
done

mkdir -p "${CHECKPOINT_ROOT}" "${LOG_ROOT}" "$(dirname "$REGISTRY_PATH")"

# =============================================================================
# IDEMPOTENCY CHECK
# =============================================================================

log_header "CHECKING FOR EXISTING LAYER ${BLOCK_LAYER} MASKED CHECKPOINTS"

for dataset in "${ALL_DATASETS[@]}"; do
    if existing=$(find_layer2_checkpoint "$dataset"); then
        echo "[SKIP] ${dataset}: layer ${BLOCK_LAYER} masked SAE already exists -> ${existing}"
        register_sae "$dataset" "$existing" "$TOTAL_TOKENS"
    else
        echo "[TRAIN] ${dataset}: no layer ${BLOCK_LAYER} masked checkpoint found"
        DATASETS_TO_TRAIN+=("$dataset")
    fi
done

if [ ${#DATASETS_TO_TRAIN[@]} -eq 0 ]; then
    log_header "NOTHING TO DO — all masked SAEs already registered"
    exit 0
fi

echo ""
echo "Training plan: ${#DATASETS_TO_TRAIN[@]} dataset(s) x ${TOTAL_TOKENS} tokens each"
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
    log_header "DATASET ${CURRENT}/${TOTAL_DATASETS}: ${dataset^^}"

    LORA_PATH=${LORA_PATHS[$dataset]}
    TRAIN_KEY=${TRAIN_DATASET_KEY[$dataset]}
    DATASET_CKPT_DIR="${CHECKPOINT_ROOT}/${dataset}"
    DATASET_LOG_DIR="${LOG_ROOT}/${dataset}"
    RUN_LOG="${DATASET_LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "${DATASET_CKPT_DIR}" "${DATASET_LOG_DIR}"

    echo "  LoRA ckpt:      ${LORA_PATH}"
    echo "  train --dataset ${TRAIN_KEY}"
    echo "  Tokens:         ${TOTAL_TOKENS}"
    echo "  Layer:          ${BLOCK_LAYER}"
    echo "  Checkpoints:    ${DATASET_CKPT_DIR}"
    echo "  Log:            ${RUN_LOG}"
    echo ""

    DATASET_START=$(date +%s)
    DATASET_START_ISO=$(date -d "@${DATASET_START}" +"%Y-%m-%d %H:%M:%S")

    python3 tasks/train_sae_masked_finetune.py \
        --sae_checkpoint_path "${SAE_CHECKPOINT}" \
        --lora_checkpoint_path "${LORA_PATH}" \
        --model_name "${MODEL_NAME}" \
        --clip_dim ${CLIP_DIM} \
        --block_layer ${BLOCK_LAYER} \
        --dataset "${TRAIN_KEY}" \
        --protect_frac ${PROTECT_FRAC} \
        --activity_n_batches ${ACTIVITY_N_BATCHES} \
        --activity_dataset "${ACTIVITY_DATASET}" \
        --expansion_factor ${EXPANSION_FACTOR} \
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
        --run_name "workshop_${dataset}_masked" \
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

log_header "DONE — ${#SUCCEEDED[@]}/${TOTAL_DATASETS} succeeded in ${OVERALL_MIN}m"
for ds in "${SUCCEEDED[@]}"; do echo "  V ${ds} -> ${CHECKPOINT_ROOT}/${ds}/"; done
for ds in "${FAILED[@]}"; do echo "  X ${ds} -> ${LOG_ROOT}/${ds}/"; done
echo ""
echo "Registry: ${REGISTRY_PATH}"
echo "Convert to --sae_paths JSON with:"
echo "  python3 tasks/registry_to_sae_paths.py --registry ${REGISTRY_PATH} --out configs/rebuttal_sae_paths.json"

if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi
