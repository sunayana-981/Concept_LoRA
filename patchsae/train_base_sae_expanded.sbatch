#!/usr/bin/env bash
#SBATCH --job-name=base_sae_expanded
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/sunayana.samavedam/Concept_LoRA/patchsae/out/rebuttal/logs/base_sae_expanded_%j.log

# 6 base SAEs (dino/align/siglip2 x topk/jumprelu) on ImageNet activations,
# at the full-scale token budget (dedicated GPU, no need for the reduced
# shared-GPU defaults used in the local sweep).
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate patchsae
cd ~/Concept_LoRA/patchsae
mkdir -p out/rebuttal/logs

export TOTAL_TOKENS_OVERRIDE=2621440
# Use the local ImageFolder (already on disk, no download) instead of the
# HF-hosted evanarlian/imagenet_1k_resized_256 dataset used by default --
# that ~24GB download exceeded this account's disk quota mid-job last time.
export DATASET_OVERRIDE=imagenet_local
bash run_train_base_sae_expanded.sh
