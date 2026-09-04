#!/usr/bin/env bash
#SBATCH --job-name=base_sae_dino_topk
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/home/sunayana.samavedam/Concept_LoRA/patchsae/out/rebuttal/logs/base_sae_dino_topk_%j.log

# One (backbone, variant) combination in isolation, run in parallel with its
# 5 siblings across separate nodes -- splitting the original single
# sequential job (6 combinations x ~11h each = ~66h) into 6 parallel jobs
# (~11h total wall-clock) since the u22 partition has no hard time limit
# and 14 nodes to spread across.
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate patchsae
cd ~/Concept_LoRA/patchsae
mkdir -p out/rebuttal/logs

export TOTAL_TOKENS_OVERRIDE=2621440
export DATASET_OVERRIDE=imagenet_local
bash run_train_base_sae_expanded.sh dino topk
