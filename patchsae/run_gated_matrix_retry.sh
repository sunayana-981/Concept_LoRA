#!/usr/bin/env bash
# Retry the gated evaluation matrix after the CUDA cache/chunking fixes.

set -uo pipefail

REPO=/home/sunayana/Documents/Concept_LoRA
PATCHSAE=$REPO/patchsae
PYTHON_BIN=/home/sunayana/miniconda3/envs/patchsae/bin/python
LOG=$PATCHSAE/out/logs/gated_matrix_retry_$(date +%Y%m%d_%H%M%S).log

mkdir -p "$PATCHSAE/out/logs"
cd "$PATCHSAE"
exec > >(tee "$LOG") 2>&1

echo "Retry queued at $(date)."
echo "Waiting for the ALIGN fine-tuning runner to finish..."
while pgrep -f '[r]un_(remaining_runnable|requested_align_only)\.sh' >/dev/null; do
  sleep 60
done

echo "ALIGN queue finished; starting gated matrix at $(date)."
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON_BIN" tasks/eval_matrix.py \
  --datasets caltech101 cityscapes cub2002011 dtd eurosat fgvc kitti officehome ucf101 \
  --vit_types base lora \
  --sae_conditions gsae_gated masked_gated \
  --sae_paths configs/rebuttal_sae_paths.json \
  --out_dir out/workshop_paper/matrix_final_gated \
  --cache_dir out/rebuttal/cache \
  --batch_size 8 \
  --max_images 1500

status=$?
echo "Gated matrix retry ended at $(date), exit=$status."
exit "$status"
