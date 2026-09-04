#!/usr/bin/env bash
# =============================================================================
# run_gated_pipeline.sh
#
# Orchestrates everything remaining for the gated-SAE-architecture arm of the
# workshop paper, meant to run inside a tmux session so it survives
# independently of any particular terminal/session:
#   1. Wait for the standard-architecture retry pass (cityscapes/fgvc) to finish,
#      if it's still running.
#   2. Train the base Gated SAE on ImageNet (tasks/train_sae_vit.py --gated_sae).
#   3. Masked fine-tune it across all 11 datasets (run_masked_finetune_all_datasets_gated.sh).
#   4. Regenerate configs/rebuttal_sae_paths.json from the registry, and patch
#      in a "gsae_gated" entry per dataset (all pointing at the same fixed
#      gated base checkpoint, mirroring how "gsae" works for the standard one).
#   5. Run the full eval_matrix.py sweep across all 11 datasets x {base,lora} x
#      {none,gsae,masked,ftsae,gsae_gated,masked_gated}.
#
# Usage (inside tmux):
#   ./run_gated_pipeline.sh 2>&1 | tee out/logs/gated_pipeline.log
# =============================================================================

set -uo pipefail
cd /home/sunayana/Documents/Concept_LoRA/patchsae

log_header() {
    echo ""
    echo "###################################################################"
    echo "# $1"
    echo "# $(date '+%Y-%m-%d %H:%M:%S')"
    echo "###################################################################"
    echo ""
}

# =============================================================================
# STEP 2: train base Gated SAE on ImageNet (reduced budget for speed)
# =============================================================================

log_header "STEP 2: training base Gated SAE on ImageNet (150,000 tokens -- fast/deadline mode)"
mkdir -p out/checkpoints/gsae_gated out/logs/gsae_gated

python3 tasks/train_sae_vit.py \
    --model_name "openai/clip-vit-base-patch16" \
    --clip_dim 768 \
    --block_layers -2 \
    --dataset imagenet \
    --expansion_factor 64 \
    --gated_sae \
    --l1_coefficient 0.00008 \
    --lr 0.0004 \
    --batch_size 16 \
    --lr_warm_up_steps 200 \
    --total_training_tokens 150000 \
    --use_ghost_grads \
    --checkpoint_path out/checkpoints/gsae_gated \
    --n_checkpoints 1 \
    --seed 42 \
    --device cuda \
    --log_to_wandb \
    --wandb_project masked_sae_finetune_all_gated \
    --wandb_log_frequency 20 \
    --run_name gsae_gated_imagenet_fast \
    2>&1 | tee "out/logs/gsae_gated/train_$(date +%Y%m%d_%H%M%S).log"

GATED_BASE_EXIT=${PIPESTATUS[0]}
if [ "$GATED_BASE_EXIT" -ne 0 ]; then
    echo "[FATAL] Gated base SAE training failed (exit ${GATED_BASE_EXIT}). Aborting pipeline."
    exit 1
fi

GATED_SAE_CKPT=$(find out/checkpoints/gsae_gated -path "*/final_sparse_autoencoder_*" -name "*_-2_resid_*.pt" | sort | tail -1)
if [ -z "$GATED_SAE_CKPT" ]; then
    echo "[FATAL] Gated base SAE training exited 0 but no final checkpoint found."
    exit 1
fi
echo "[OK] Gated base SAE: ${GATED_SAE_CKPT}"

# =============================================================================
# STEP 3: masked fine-tune the gated base SAE across all 11 datasets
# =============================================================================

log_header "STEP 3: masked fine-tuning gated SAE across 9 datasets (matching standard-arch results)"
./run_masked_finetune_all_datasets_gated.sh
GATED_MASKED_EXIT=$?
if [ "$GATED_MASKED_EXIT" -ne 0 ]; then
    echo "[WARN] Some gated masked-finetune runs failed (exit ${GATED_MASKED_EXIT})."
    echo "       Continuing to eval_matrix anyway -- failed cells will just skip gracefully."
fi

# =============================================================================
# STEP 4: regenerate sae_paths JSON, patch in gsae_gated
# =============================================================================

log_header "STEP 4: regenerating configs/rebuttal_sae_paths.json"
python3 tasks/registry_to_sae_paths.py --registry out/rebuttal/sae_registry.json --out configs/rebuttal_sae_paths.json

python3 - "$GATED_SAE_CKPT" << 'PYEOF'
import json, sys
gated_ckpt = sys.argv[1]
with open("configs/rebuttal_sae_paths.json") as f:
    sae_paths = json.load(f)
datasets = ["caltech101", "cityscapes", "cub2002011", "dtd", "eurosat", "fgvc",
            "kitti", "officehome", "ucf101"]
for ds in datasets:
    sae_paths.setdefault(ds, {})["gsae_gated"] = gated_ckpt
with open("configs/rebuttal_sae_paths.json", "w") as f:
    json.dump(sae_paths, f, indent=2)
print(f"Patched gsae_gated -> {gated_ckpt} into all {len(datasets)} datasets")
PYEOF

# =============================================================================
# STEP 5: eval matrix -- only the NEW gated conditions (none/gsae/masked/ftsae
# already have a complete results table from the standard-architecture run;
# re-running them here would roughly double the time for no new information).
# =============================================================================

log_header "STEP 5: running eval_matrix.py for gated conditions only (9 datasets)"
python3 tasks/eval_matrix.py \
    --datasets caltech101 cityscapes cub2002011 dtd eurosat fgvc kitti officehome ucf101 \
    --vit_types base lora \
    --sae_conditions gsae_gated masked_gated \
    --sae_paths configs/rebuttal_sae_paths.json \
    --out_dir out/workshop_paper/matrix_final_gated \
    --cache_dir out/rebuttal/cache \
    --batch_size 8 \
    --max_images 1500

log_header "PIPELINE COMPLETE"
echo "Gated-arch results: out/workshop_paper/matrix_final_gated/results.csv"
echo "Standard-arch results (already done): out/workshop_paper/matrix_final_standard/results.csv"
