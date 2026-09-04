#!/usr/bin/env bash
#SBATCH --job-name=masked_finetune_expanded
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/sunayana.samavedam/Concept_LoRA/patchsae/out/rebuttal/logs/masked_finetune_expanded_%j.log

# 54 masked-finetune runs (dino/align/siglip2 x topk/jumprelu x 9 datasets),
# at the full-scale token budget. Depends on train_base_sae_expanded.sbatch's
# checkpoints existing first -- submit with:
#   sbatch --dependency=afterok:<base_sae_job_id> masked_finetune_expanded.sbatch
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate patchsae
cd ~/Concept_LoRA/patchsae
mkdir -p out/rebuttal/logs

export TOTAL_TOKENS_OVERRIDE=2621440
export ACTIVITY_DATASET_OVERRIDE=imagenet_local
bash run_masked_finetune_expanded.sh
