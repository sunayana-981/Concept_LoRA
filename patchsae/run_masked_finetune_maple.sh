#!/usr/bin/env bash
# =============================================================================
# run_masked_finetune_maple.sh
#
# Masked fine-tune of a pre-trained ImageNet SAE on MaPLe-adapted CLIP activations for MedMNIST.
#
# Usage:
#   chmod +x run_masked_finetune_maple.sh
#   ./run_masked_finetune_maple.sh 2>&1 | tee masked_finetune_maple_log.txt
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION — EDIT THESE
# =============================================================================

PROJECT_ROOT="/home/sunayana/Documents/Concept_LoRA/patchsae"
CHECKPOINT_ROOT="/home/sunayana/Documents/Concept_LoRA/patchsae/out/checkpoints/masked_finetune_maple"
LOG_ROOT="/home/sunayana/Documents/Concept_LoRA/patchsae/out/logs/masked_finetune_maple"

# --- Pre-trained SAE checkpoint (from previous MaPLe run) ---
SAE_CHECKPOINT="/home/sunayana/Documents/Concept_LoRA/patchsae/data/sae_weight/base/out.pt"

# --- MaPLe checkpoint and config ---
MAPLE_MODEL_PATH="/home/sunayana/Documents/model.pth.tar-5"
MAPLE_CONFIG_PATH="/home/sunayana/Documents/Concept_LoRA/patchsae/configs/models/maple/vit_b16_c2_ep5_batch4_2ctx.yaml"

# --- Dataset ---
DATASET="medmnist"

# --- Hyperparameters ---
PROTECT_FRAC=0.2
ACTIVITY_N_BATCHES=50
L1_COEFFICIENT=0.00008
BATCH_SIZE=16
TOTAL_TOKENS=1000000
SEED=42
DEVICE="cuda"

# --- W&B ---
WANDB_PROJECT="masked_sae_finetune_maple"
WANDB_LOG_FREQ=20

mkdir -p "$CHECKPOINT_ROOT" "$LOG_ROOT"

# =============================================================================
# RUN
# =============================================================================

python3 tasks/train_sae_masked_finetune.py \
    --sae_checkpoint_path "$SAE_CHECKPOINT" \
    --vit_type maple \
    --model_path "$MAPLE_MODEL_PATH" \
    --config_path "$MAPLE_CONFIG_PATH" \
    --dataset "$DATASET" \
    --protect_frac $PROTECT_FRAC \
    --activity_n_batches $ACTIVITY_N_BATCHES \
    --l1_coefficient $L1_COEFFICIENT \
    --batch_size $BATCH_SIZE \
    --total_training_tokens $TOTAL_TOKENS \
    --checkpoint_path "$CHECKPOINT_ROOT" \
    --seed $SEED \
    --device $DEVICE \
    --log_to_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_log_frequency $WANDB_LOG_FREQ \
    2>&1 | tee "$LOG_ROOT/train_$(date +%Y%m%d_%H%M%S).log"
