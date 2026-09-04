#!/usr/bin/env bash
#SBATCH --job-name=maple_sweep
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/home/sunayana.samavedam/Concept_LoRA/unified_logs/maple_sweep_%j.log

# 16 MaPLe runs (clip x 16 datasets), at the official recipe's epochs=5
# (full scale, not the reduced epochs=3 used in the local shared-GPU
# sweep). Independent of the other jobs -- can run in parallel.
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate patchsae
cd ~/Concept_LoRA
mkdir -p unified_logs

export EPOCHS_OVERRIDE=5
bash run_all_maple.sh
