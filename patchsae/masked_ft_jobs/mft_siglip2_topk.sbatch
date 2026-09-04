#!/usr/bin/env bash
#SBATCH --job-name=mft_siglip2_topk
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/sunayana.samavedam/Concept_LoRA/patchsae/out/rebuttal/logs/mft_siglip2_topk_%j.log

# Masked fine-tune of one (backbone, variant) base SAE across all 9 datasets.
#
# TOTAL_TOKENS matches this repo's own masked-finetune convention
# (run_masked_finetune_all_datasets.sh uses 100000; the gated variant uses
# 30000) -- NOT the 2,621,440 used for training base SAEs from scratch. This
# step adapts an already-trained dictionary, so it needs a fraction of the
# budget: ~25min/dataset here versus ~11h at the base-SAE budget.
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate patchsae
cd ~/Concept_LoRA/patchsae
mkdir -p out/rebuttal/logs

export TOTAL_TOKENS_OVERRIDE=100000
export ACTIVITY_DATASET_OVERRIDE=imagenet_local
bash run_masked_finetune_expanded.sh siglip2 topk
