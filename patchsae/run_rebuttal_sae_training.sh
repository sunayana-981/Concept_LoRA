#!/usr/bin/env bash
# =============================================================================
# run_rebuttal_sae_training.sh
#
# Trains random-init target-domain Scratch-SAEs ("pets" and "eurosat"), extending
# run_all_sae_training.sh's pattern to this repo's own LoRA-CLIP SAE trainer
# (tasks/train_sae_lora_clip.py), fixed at the rebuttal's non-negotiable
# hyperparameters (layer -2, expansion factor 64, L1 8e-5, seed 42 -- see
# tasks/rebuttal_common.py / configs/rebuttal_datasets.json for the same
# convention used by the eval drivers).
#
# Idempotent: a dataset is skipped if a FINAL layer -2 checkpoint already
# exists under out/checkpoints/{dataset}/. NOTE: out/checkpoints/eurosat/
# already has two completed runs from earlier work, but at block_layer -1 and
# -3 -- NOT -2 -- so they do NOT satisfy the rebuttal convention and eurosat
# will be retrained.
#
# Registers every (dataset, "scratchsae") checkpoint in out/rebuttal/sae_registry.json,
# which tasks/registry_to_sae_paths.py turns into the --sae_paths JSON that
# tasks/eval_matrix.py / tasks/eval_steering.py consume.
#
# Usage:
#   chmod +x run_rebuttal_sae_training.sh
#   nohup ./run_rebuttal_sae_training.sh > out/rebuttal/logs/sae_training.log 2>&1 &
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT="/home/sunayana/Documents/Concept_LoRA/patchsae"
LORA_WEIGHTS_ROOT="/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16"
CHECKPOINT_ROOT="${PROJECT_ROOT}/out/checkpoints"
LOG_ROOT="${PROJECT_ROOT}/out/logs/rebuttal"
REGISTRY_PATH="${PROJECT_ROOT}/out/rebuttal/sae_registry.json"

# --- rebuttal dataset name -> LoRA checkpoint path ---
declare -A LORA_PATHS
LORA_PATHS["eurosat"]="${LORA_WEIGHTS_ROOT}/eurosat/16shots/seed1/lora_weights.pt"
LORA_PATHS["pets"]="${LORA_WEIGHTS_ROOT}/pets/16shots/seed1/lora_weights.pt"

# --- rebuttal dataset name -> tasks/utils.py DATASET_INFO key ---
# ("pets" is the rebuttal-facing name used throughout tasks/*.py and
#  configs/rebuttal_datasets.json; the repo's DATASET_INFO / lora_weights
#  directory both call it "oxford_pets".)
declare -A TRAIN_DATASET_KEY
TRAIN_DATASET_KEY["eurosat"]="eurosat"
TRAIN_DATASET_KEY["pets"]="oxford_pets"

REBUTTAL_DATASETS=(eurosat pets)

# --- Fixed rebuttal hyperparameters (never vary across conditions) ---
BLOCK_LAYER=-2
EXPANSION_FACTOR=64
L1_COEFFICIENT=0.00008
LR=0.0004
BATCH_SIZE=16
LR_WARM_UP_STEPS=500
CLIP_DIM=768
MODEL_NAME="openai/clip-vit-base-patch16"
SEED=42
TOTAL_TOKENS=2000000
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

# Find an existing FINAL checkpoint at BLOCK_LAYER for a dataset, if any.
# Matches the naming pattern this repo's SAE trainer already uses, e.g.
# out/checkpoints/eurosat/<run_id>/final_sparse_autoencoder_openai/clip-vit-base-patch16_-2_resid_49152.pt
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
    local dataset=$1 ckpt=$2 examples=$3
    local lora_path=${LORA_PATHS[$dataset]}
    python3 - "$REGISTRY_PATH" "$dataset" "$ckpt" "$examples" "$BLOCK_LAYER" "$lora_path" <<'PYEOF'
import json, os, sys, time
registry_path, dataset, ckpt, examples, layer, lora_path = sys.argv[1:7]
os.makedirs(os.path.dirname(registry_path), exist_ok=True)
records = []
if os.path.exists(registry_path):
    with open(registry_path) as f:
        records = json.load(f)
records = [r for r in records if not (r["dataset"] == dataset and r["condition"] == "scratchsae")]
records.append({
    "dataset": dataset, "vit_type": "lora", "condition": "scratchsae",
    "checkpoint_path": ckpt, "training_examples": int(examples), "layer": int(layer),
    "activation_vectors_per_example": 197,
    "derived_activation_vector_exposure_requested": int(examples) * 197,
    "sae_initialization": "scratch_random", "initialization_checkpoint": None,
    "activation_data_role": "target", "training_seed": 42,
    "architecture": "standard", "adapted_model_checkpoint": lora_path,
    "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
})
with open(registry_path, "w") as f:
    json.dump(records, f, indent=2)
print(f"[REGISTRY] {dataset}/scratchsae -> {ckpt}")
PYEOF
}

# =============================================================================
# PRE-FLIGHT
# =============================================================================

log_header "PRE-FLIGHT CHECKS"
cd "$PROJECT_ROOT"

if [ ! -f "tasks/train_sae_lora_clip.py" ]; then
    echo "[FATAL] tasks/train_sae_lora_clip.py not found"; exit 1
fi
echo "Python: $(python3 --version 2>&1)"
nvidia-smi --query-gpu=name,memory.free,temperature.gpu --format=csv,noheader 2>/dev/null || true

# Resolve "pets": brief-specified path vs this repo's established alias.
PETS_ALIAS_PATH="${LORA_WEIGHTS_ROOT}/oxford_pets/16shots/seed1/lora_weights.pt"
if [ ! -f "${LORA_PATHS[pets]}" ]; then
    if [ -f "${PETS_ALIAS_PATH}" ]; then
        echo "[NOTE] ${LORA_PATHS[pets]} not found, but the equivalent checkpoint exists"
        echo "       under this repo's established directory name:"
        echo "         ${PETS_ALIAS_PATH}"
        echo "       (tasks/utils.py's DATASET_INFO key is 'oxford_pets'; the rebuttal"
        echo "        dataset registry configs/rebuttal_datasets.json already points"
        echo "        'pets' -> this same checkpoint.) Using it."
        LORA_PATHS["pets"]="${PETS_ALIAS_PATH}"
    else
        echo "[FATAL] No LoRA checkpoint found for 'pets' at either:"
        echo "          ${LORA_PATHS[pets]}"
        echo "          ${PETS_ALIAS_PATH}"
        echo ""
        echo "Run this to produce one (adjust --root_path to your CLIP_LoRA data root):"
        echo "  cd /home/sunayana/Documents/Concept_LoRA && python lora_main.py \\"
        echo "    --root_path <DATA_ROOT> --dataset oxford_pets --shots 16 --seed 1 \\"
        echo "    --backbone ViT-B/16 --r 2 --alpha 1 --encoder both --params q k v \\"
        echo "    --save_path ${LORA_WEIGHTS_ROOT}/pets/16shots/seed1 --filename lora_weights"
        exit 1
    fi
fi

for dataset in "${REBUTTAL_DATASETS[@]}"; do
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

log_header "CHECKING FOR EXISTING LAYER ${BLOCK_LAYER} CHECKPOINTS"

DATASETS_TO_TRAIN=()
for dataset in "${REBUTTAL_DATASETS[@]}"; do
    if existing=$(find_layer2_checkpoint "$dataset"); then
        echo "[SKIP] ${dataset}: layer ${BLOCK_LAYER} Scratch-SAE already exists -> ${existing}"
        register_sae "$dataset" "$existing" "$TOTAL_TOKENS"
    else
        echo "[TRAIN] ${dataset}: no layer ${BLOCK_LAYER} checkpoint found"
        DATASETS_TO_TRAIN+=("$dataset")
    fi
done

if [ ${#DATASETS_TO_TRAIN[@]} -eq 0 ]; then
    log_header "NOTHING TO DO — all Scratch-SAEs already registered"
    exit 0
fi

est_min_each=$(( TOTAL_TOKENS / 91 / 60 ))
echo ""
echo "Training plan (~91 it/s reference from prior EuroSAT run):"
for ds in "${DATASETS_TO_TRAIN[@]}"; do
    echo "  ${ds}: ${TOTAL_TOKENS} training examples -> ~${est_min_each} min"
done
echo "  TOTAL: ~$(( est_min_each * ${#DATASETS_TO_TRAIN[@]} )) min"

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

    set +e
    python3 tasks/train_sae_lora_clip.py \
        --model_name "${MODEL_NAME}" \
        --clip_dim ${CLIP_DIM} \
        --lora_checkpoint_path "${LORA_PATH}" \
        --sae_initialization scratch \
        --sae_condition scratchsae \
        --target_dataset "${dataset}" \
        --activation_data_role target \
        --protect_frac 0 \
        --block_layers ${BLOCK_LAYER} \
        --dataset "${TRAIN_KEY}" \
        --expansion_factor ${EXPANSION_FACTOR} \
        --l1_coefficient ${L1_COEFFICIENT} \
        --lr ${LR} \
        --batch_size ${BATCH_SIZE} \
        --lr_warm_up_steps ${LR_WARM_UP_STEPS} \
        --training_examples ${TOTAL_TOKENS} \
        --use_ghost_grads \
        --checkpoint_path "${DATASET_CKPT_DIR}" \
        --n_checkpoints 3 \
        --seed ${SEED} \
        --device ${DEVICE} \
        --log_to_wandb \
        --wandb_project "${WANDB_PROJECT}" \
        --wandb_log_frequency ${WANDB_LOG_FREQ} \
        --run_name "rebuttal_${dataset}_scratchsae" \
        2>&1 | tee "${RUN_LOG}"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

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
