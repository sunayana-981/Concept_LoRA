#!/usr/bin/env bash
# =============================================================================
# run_new_datasets.sh
#
# Full pipeline for officehome, kitti, and cityscapes:
#   1. Convert raw data → ImageFolder
#   2. LoRA fine-tune CLIP ViT-B/16 (16-shot, seed 1)
#   3. Train SAEs on base CLIP activations
#   4. Train SAEs on LoRA-adapted activations (layers -1 and -2)
#
# Usage:
#   bash run_new_datasets.sh 2>&1 | tee new_datasets_training.log
#
# Background (SSH-safe):
#   nohup bash run_new_datasets.sh > new_datasets_training.log 2>&1 &
# =============================================================================

set -uo pipefail

REPO=/home/sunayana/Documents/Concept_LoRA
DATA=$REPO/data
PATCHSAE=$REPO/patchsae
LORA_ROOT=$REPO/lora_weights/vitb16
WANDB_PROJECT="lora_clip_sae"

export CUDA_VISIBLE_DEVICES=0

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

# =============================================================================
# STEP 1 — Data preparation
# =============================================================================
echo ""
echo "================================================================="
echo "STEP 1: Data preparation  [$(timestamp)]"
echo "================================================================="

echo ""
echo "--- OfficeHome: building merged imagefolder (symlinks) ---"
python "$PATCHSAE/../sae_vlm/tasks/setup_officehome_imagefolder.py" \
    --src "$DATA/OfficeHomeDataset_10072016" \
    --dst "$DATA/officehome_imagefolder"

echo ""
echo "--- KITTI: converting parquet → ImageFolder ---"
python "$PATCHSAE/../sae_vlm/tasks/convert_kitti_to_imagefolder.py" \
    --kitti_root "$DATA/kitti/data" \
    --out_root   "$DATA/kitti_imagefolder"

echo ""
echo "--- Cityscapes: converting parquet → ImageFolder ---"
python "$PATCHSAE/../sae_vlm/tasks/convert_cityscapes_to_imagefolder.py" \
    --cityscapes_root "$DATA/cityscapes/data" \
    --out_root        "$DATA/cityscapes_imagefolder"

# =============================================================================
# STEP 2 — LoRA fine-tuning
# =============================================================================
echo ""
echo "================================================================="
echo "STEP 2: LoRA fine-tuning (16-shot, seed 1)  [$(timestamp)]"
echo "================================================================="

lora_train() {
    local dataset=$1
    local root=$2
    echo ""
    echo "--- LoRA fine-tune: $dataset ---"
    python "$REPO/unified_finetune.py" \
        --model      clip \
        --method     lora \
        --dataset    "$dataset" \
        --root_path  "$root" \
        --backbone   "ViT-B/16" \
        --shots      16 \
        --n_iters    100 \
        --lr         2e-4 \
        --r          4 \
        --alpha      1 \
        --position   all \
        --encoder    both \
        --params     q k v \
        --batch_size 32 \
        --seed       1 \
        --save_path  "$REPO/lora_weights" \
        --filename   "lora_weights" \
        --no_linear_probe
    echo "  → weights: $LORA_ROOT/$dataset/16shots/seed1/lora_weights.pt"
}

lora_train officehome "$DATA"
lora_train kitti      "$DATA"
lora_train cityscapes "$DATA"

# =============================================================================
# STEP 3 — SAE training
# =============================================================================
echo ""
echo "================================================================="
echo "STEP 3: SAE training  [$(timestamp)]"
echo "================================================================="

# SAE training always runs from patchsae/ directory
cd "$PATCHSAE"

sae_train() {
    local dataset=$1
    local layer=$2
    local tokens=$3
    local lora_path=${4:-""}          # empty = base CLIP
    local run_name=$5

    echo ""
    if [ -z "$lora_path" ]; then
        echo "--- SAE (base CLIP): $dataset  layer=$layer ---"
    else
        echo "--- SAE (LoRA CLIP): $dataset  layer=$layer  lora=$lora_path ---"
    fi

    local extra=""
    if [ -n "$lora_path" ]; then
        extra="--lora_checkpoint_path $lora_path"
    fi

    python tasks/train_sae_lora_clip.py \
        --model_name      "openai/clip-vit-base-patch16" \
        --clip_dim        768 \
        --block_layers    "$layer" \
        --dataset         "$dataset" \
        --expansion_factor 64 \
        --l1_coefficient  0.00008 \
        --lr              0.0004 \
        --batch_size      16 \
        --lr_warm_up_steps 500 \
        --total_training_tokens "$tokens" \
        --use_ghost_grads \
        --checkpoint_path "out/checkpoints/$dataset" \
        --n_checkpoints   3 \
        --seed            42 \
        --device          cuda \
        --log_to_wandb \
        --wandb_project   "$WANDB_PROJECT" \
        --wandb_log_frequency 20 \
        --run_name        "$run_name" \
        $extra
}

# Token budgets
OH_TOKENS=200000
KT_TOKENS=200000
CS_TOKENS=200000

LORA_OH="$LORA_ROOT/officehome/16shots/seed1/lora_weights.pt"
LORA_KT="$LORA_ROOT/kitti/16shots/seed1/lora_weights.pt"
LORA_CS="$LORA_ROOT/cityscapes/16shots/seed1/lora_weights.pt"

# ── Base SAEs (no LoRA) ────────────────────────────────────────────────────────
sae_train officehome -2 $OH_TOKENS "" "officehome_base_l-2"
sae_train kitti      -2 $KT_TOKENS "" "kitti_base_l-2"
sae_train cityscapes -2 $CS_TOKENS "" "cityscapes_base_l-2"

# ── LoRA SAEs ─────────────────────────────────────────────────────────────────
sae_train officehome -2 $OH_TOKENS "$LORA_OH" "officehome_lora_l-2"
sae_train officehome -1 $OH_TOKENS "$LORA_OH" "officehome_lora_l-1"

sae_train kitti      -2 $KT_TOKENS "$LORA_KT" "kitti_lora_l-2"
sae_train kitti      -1 $KT_TOKENS "$LORA_KT" "kitti_lora_l-1"

sae_train cityscapes -2 $CS_TOKENS "$LORA_CS" "cityscapes_lora_l-2"
sae_train cityscapes -1 $CS_TOKENS "$LORA_CS" "cityscapes_lora_l-1"

echo ""
echo "================================================================="
echo "All done.  [$(timestamp)]"
echo ""
echo "SAE checkpoints:"
find "$PATCHSAE/out/checkpoints/officehome" \
     "$PATCHSAE/out/checkpoints/kitti" \
     "$PATCHSAE/out/checkpoints/cityscapes" \
     -name "*.pt" 2>/dev/null | head -20
echo ""
echo "Next: update RUNS in sae_vlm/sweep_dams_hyperparams.py"
echo "with the wandb run IDs printed during SAE training."
echo "================================================================="
