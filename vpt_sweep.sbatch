#!/usr/bin/env bash
#SBATCH --job-name=vpt_sweep
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/sunayana.samavedam/Concept_LoRA/unified_logs/vpt_sweep_%j.log

# 48 VPT runs (clip/dino/siglip2 x 16 datasets), at the official recipe's
# n_iters=100 (full scale, not the reduced n_iters=10 used in the local
# shared-GPU sweep). Independent of the SAE jobs -- can run in parallel.
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate patchsae
cd ~/Concept_LoRA
mkdir -p unified_logs

export N_ITERS_OVERRIDE=100
bash run_all_vpt.sh
