#!/usr/bin/env bash
# =============================================================================
# run_fullft_sae_training.sh
#
# Trains the ONE FullFT-SAE baseline for the rebuttal: an SAE trained on the
# PathMNIST-LoRA-adapted CLIP (the "medmnist" LoRA checkpoint) using plain
# ImageNet images as the activation source -- the "standard practice" of
# training an SAE on generic data even when the backbone was fine-tuned on a
# specific target domain. Same fixed hyperparameters as Task 4's FT-SAEs
# (layer -2, expansion factor 64, L1 8e-5, seed 42) except the token budget,
# which is 10M here vs. 2M for the target-domain FT-SAEs.
#
# Registers the checkpoint in out/rebuttal/sae_registry.json under dataset
# "pathmnist", condition "fullftsae" (matching configs/rebuttal_datasets.json's
# "pathmnist" naming, not the repo's internal "medmnist" DATASET_INFO key).
#
# Usage:
#   chmod +x run_fullft_sae_training.sh
#   nohup ./run_fullft_sae_training.sh --imagenet_train_dir /path/to/imagenet/train \
#       > out/rebuttal/logs/fullft_sae_training.log 2>&1 &
# =============================================================================

set -euo pipefail

# =============================================================================
# CLI
# =============================================================================

IMAGENET_TRAIN_DIR=""
while [ $# -gt 0 ]; do
    case "$1" in
        --imagenet_train_dir=*) IMAGENET_TRAIN_DIR="${1#*=}"; shift ;;
        --imagenet_train_dir) IMAGENET_TRAIN_DIR="$2"; shift 2 ;;
        *) echo "[FATAL] Unknown argument: $1"; exit 1 ;;
    esac
done

if [ -z "$IMAGENET_TRAIN_DIR" ]; then
    echo "[FATAL] --imagenet_train_dir <path> is required (local ImageNet train"
    echo "        directory in ImageFolder layout: <dir>/<class>/<image>.jpg)."
    echo ""
    echo "Usage: ./run_fullft_sae_training.sh --imagenet_train_dir /path/to/imagenet/train"
    exit 1
fi
if [ ! -d "$IMAGENET_TRAIN_DIR" ]; then
    echo "[FATAL] --imagenet_train_dir does not exist or is not a directory: ${IMAGENET_TRAIN_DIR}"
    exit 1
fi
export IMAGENET_TRAIN_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT="/home/sunayana/Documents/Concept_LoRA/patchsae"
LORA_WEIGHTS_ROOT="/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16"
CHECKPOINT_ROOT="${PROJECT_ROOT}/out/checkpoints"
LOG_ROOT="${PROJECT_ROOT}/out/logs/rebuttal"
REGISTRY_PATH="${PROJECT_ROOT}/out/rebuttal/sae_registry.json"

REBUTTAL_DATASET="pathmnist"          # rebuttal-facing name (configs/rebuttal_datasets.json)
TRAIN_DATASET_KEY="imagenet_local"    # tasks/utils.py DATASET_INFO key (activation source)
LORA_PATH="${LORA_WEIGHTS_ROOT}/medmnist/16shots/seed1/lora_weights.pt"   # PathMNIST LoRA ckpt
DATASET_CKPT_DIR="${CHECKPOINT_ROOT}/pathmnist_fullft"

# --- Fixed hyperparameters (identical to Task 4 except token budget) ---
BLOCK_LAYER=-2
EXPANSION_FACTOR=64
L1_COEFFICIENT=0.00008
LR=0.0004
BATCH_SIZE=16
LR_WARM_UP_STEPS=500
CLIP_DIM=768
MODEL_NAME="openai/clip-vit-base-patch16"
SEED=42
TOTAL_TOKENS=10000000
WANDB_PROJECT="lora_clip_sae"
WANDB_LOG_FREQ=20
DEVICE="cuda"
export CUDA_VISIBLE_DEVICES=0

# Empirically observed throughput (from Task 4's eurosat FT-SAE run: tqdm
# reported tokens/sec 1:1 with "it/s", NOT multiplied by batch_size).
OBSERVED_TOKENS_PER_SEC=91

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
    shopt -s nullglob
    local matches=("${DATASET_CKPT_DIR}"/*/final_sparse_autoencoder_*/*_${BLOCK_LAYER}_resid_*.pt)
    shopt -u nullglob
    if [ ${#matches[@]} -gt 0 ]; then
        printf '%s\n' "${matches[-1]}"
        return 0
    fi
    return 1
}

register_sae() {
    local ckpt=$1 tokens=$2
    python3 - "$REGISTRY_PATH" "$REBUTTAL_DATASET" "$ckpt" "$tokens" "$BLOCK_LAYER" <<'PYEOF'
import json, os, sys, time
registry_path, dataset, ckpt, tokens, layer = sys.argv[1:6]
os.makedirs(os.path.dirname(registry_path), exist_ok=True)
records = []
if os.path.exists(registry_path):
    with open(registry_path) as f:
        records = json.load(f)
records = [r for r in records if not (r["dataset"] == dataset and r["condition"] == "fullftsae")]
records.append({
    "dataset": dataset, "vit_type": "lora", "condition": "fullftsae",
    "checkpoint_path": ckpt, "training_examples": int(tokens), "layer": int(layer),
    "activation_vectors_per_example": 197,
    "derived_activation_vector_exposure_requested": int(tokens) * 197,
    "sae_initialization": "scratch_random", "initialization_checkpoint": None,
    "activation_dataset": "imagenet_local", "activation_data_role": "generic",
    "target_dataset": "pathmnist", "training_seed": 42,
    "architecture": "standard",
    "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
})
with open(registry_path, "w") as f:
    json.dump(records, f, indent=2)
print(f"[REGISTRY] {dataset}/fullftsae -> {ckpt}")
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
if [ ! -f "$LORA_PATH" ]; then
    echo "[FATAL] PathMNIST LoRA checkpoint not found: ${LORA_PATH}"; exit 1
fi
echo "[OK] LoRA checkpoint: ${LORA_PATH}"
echo "[OK] ImageNet train dir: ${IMAGENET_TRAIN_DIR}"
N_CLASS_DIRS=$(find "${IMAGENET_TRAIN_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "     (${N_CLASS_DIRS} class subdirectories found)"
echo "Python: $(python3 --version 2>&1)"
nvidia-smi --query-gpu=name,memory.free,temperature.gpu --format=csv,noheader 2>/dev/null || true

mkdir -p "${DATASET_CKPT_DIR}" "${LOG_ROOT}/pathmnist_fullft" "$(dirname "$REGISTRY_PATH")"

# =============================================================================
# IDEMPOTENCY CHECK
# =============================================================================

log_header "CHECKING FOR EXISTING LAYER ${BLOCK_LAYER} FULLFT-SAE"

if existing=$(find_layer2_checkpoint); then
    echo "[SKIP] pathmnist/fullftsae: layer ${BLOCK_LAYER} checkpoint already exists -> ${existing}"
    register_sae "$existing" "$TOTAL_TOKENS"
    echo ""
    echo "Nothing to do. Delete ${DATASET_CKPT_DIR} to force retraining."
    exit 0
fi

# =============================================================================
# ETA
# =============================================================================

est_sec=$(( TOTAL_TOKENS / OBSERVED_TOKENS_PER_SEC ))
est_hr=$(( est_sec / 3600 ))
est_min=$(( (est_sec % 3600) / 60 ))

log_header "ETA ESTIMATE"
echo "  Token budget:            ${TOTAL_TOKENS}"
echo "  Observed rate:           ~${OBSERVED_TOKENS_PER_SEC} tokens/s"
echo "                            (measured from Task 4's eurosat FT-SAE run;"
echo "                             this is tokens/s, not batches/s -- the SAE"
echo "                             trainer's tqdm 'it' already denotes 1 token.)"
echo "  Estimated wall time:      ~${est_hr}h ${est_min}m"
echo ""
echo "  NOTE: this is well above the ~4-6h originally planned for this run."
echo "  At the same throughput as the 2M-token FT-SAEs (~6h/run), a 10M-token"
echo "  budget scales to roughly 5x that: expect ~30h, not ~5h. Consider"
echo "  reducing --total_training_tokens below if you need this to fit an"
echo "  overnight window; 10M is kept here because the brief asked for it."
echo ""

# =============================================================================
# TRAINING
# =============================================================================

log_header "TRAINING pathmnist FULLFT-SAE (ImageNet activations)"

RUN_LOG="${LOG_ROOT}/pathmnist_fullft/train_$(date +%Y%m%d_%H%M%S).log"
echo "  Log: ${RUN_LOG}"
echo ""

START=$(date +%s)
START_ISO=$(date -d "@${START}" +"%Y-%m-%d %H:%M:%S")

set +e
python3 tasks/train_sae_lora_clip.py \
    --model_name "${MODEL_NAME}" \
    --clip_dim ${CLIP_DIM} \
    --lora_checkpoint_path "${LORA_PATH}" \
    --sae_initialization scratch \
    --sae_condition fullftsae \
    --target_dataset pathmnist \
    --activation_data_role generic \
    --protect_frac 0 \
    --block_layers ${BLOCK_LAYER} \
    --dataset "${TRAIN_DATASET_KEY}" \
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
    --run_name "rebuttal_pathmnist_fullftsae" \
    2>&1 | tee "${RUN_LOG}"
EXIT_CODE=${PIPESTATUS[0]}
set -e

END=$(date +%s)
MIN=$(( (END - START) / 60 ))

if [ ${EXIT_CODE} -ne 0 ]; then
    echo "[FAILED] pathmnist fullftsae (exit ${EXIT_CODE}) after ${MIN}m — see ${RUN_LOG}"
    exit 1
fi

NEW_CKPT=$(find "${DATASET_CKPT_DIR}" -newermt "${START_ISO}" \
    -path "*/final_sparse_autoencoder_*" -name "*_${BLOCK_LAYER}_resid_*.pt" 2>/dev/null \
    | sort | tail -1)

if [ -z "${NEW_CKPT}" ]; then
    echo "[WARN] training exited 0 but no final layer ${BLOCK_LAYER} checkpoint was"
    echo "       found under ${DATASET_CKPT_DIR} — not registered."
    exit 1
fi

echo "[SUCCESS] pathmnist fullftsae done in ${MIN}m -> ${NEW_CKPT}"
register_sae "$NEW_CKPT" "$TOTAL_TOKENS"

log_header "DONE"
echo "Registry: ${REGISTRY_PATH}"
echo "Convert to --sae_paths JSON with:"
echo "  python3 tasks/registry_to_sae_paths.py --registry ${REGISTRY_PATH} --out configs/rebuttal_sae_paths.json"
