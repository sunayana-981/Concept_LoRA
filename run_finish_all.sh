#!/usr/bin/env bash
# Finish every missing/corrected LoRA + DoRA cell, then retry gated evaluation.

set -uo pipefail

REPO=/home/sunayana/Documents/Concept_LoRA
DATA=$REPO/data
SAVE=$REPO/unified_weights
LOG_DIR=$REPO/unified_logs_finish_all
RESULTS=$REPO/results/accuracy_results.csv
SUMMARY=$REPO/results/finish_all_summary.tsv
PATCHSAE_PY=/home/sunayana/miniconda3/envs/patchsae/bin/python
SIGLIP_PY=/home/sunayana/miniconda3/envs/dncbm310/bin/python
N_ITERS=${N_ITERS:-100}

mkdir -p "$LOG_DIR" "$REPO/results"
touch "$SUMMARY"
cd "$REPO"

if [[ ! -s "$SUMMARY" ]]; then
  printf 'status\tmodel\tmethod\tdataset\tdetail\n' > "$SUMMARY"
fi

checkpoint_path() {
  local model=$1 method=$2 dataset=$3
  case "$model" in
    clip)   printf '%s/vitb16/%s/16shots/seed1/%s_adapter_weights.pt' "$SAVE" "$dataset" "$method" ;;
    dino)   printf '%s/dino_dinov2-base/%s/16shots/seed1/%s_adapter_weights.pt' "$SAVE" "$dataset" "$method" ;;
    align)  printf '%s/align-base/%s/16shots/seed1/%s_adapter_weights.pt' "$SAVE" "$dataset" "$method" ;;
    siglip) printf '%s/siglip2-base-patch16-224/%s/16shots/seed1/%s_adapter_weights.pt' "$SAVE" "$dataset" "$method" ;;
  esac
}

run_one() {
  local model=$1 dataset=$2 root=$3 batch=$4 encoder=$5 method=$6
  local python_bin=$PATCHSAE_PY
  [[ "$model" == siglip ]] && python_bin=$SIGLIP_PY

  local checkpoint
  checkpoint=$(checkpoint_path "$model" "$method" "$dataset")
  if [[ -s "$checkpoint" && "${FORCE_RERUN:-0}" != 1 ]]; then
    printf 'EXISTS\t%s\t%s\t%s\t%s\n' "$model" "$method" "$dataset" "$checkpoint" >> "$SUMMARY"
    echo "EXISTS $model/$method/$dataset"
    return 0
  fi

  local log=$LOG_DIR/${model}_${method}_${dataset}.log
  local attempt_batch=$batch
  local attempt=1
  while (( attempt <= 2 )); do
    echo "RUN $model/$method/$dataset batch=$attempt_batch attempt=$attempt"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false \
      "$python_bin" unified_finetune.py \
        --model "$model" --method "$method" --dataset "$dataset" \
        --root_path "$root" --backbone ViT-B/16 --shots 16 \
        --n_iters "$N_ITERS" --lr 2e-4 --r 4 --alpha 1 \
        --position all --encoder "$encoder" --params q k v \
        --batch_size "$attempt_batch" --seed 1 --save_path "$SAVE" \
        --filename "${method}_adapter_weights" --results_csv "$RESULTS" \
        --no_linear_probe 2>&1 | tee "$log"
    local exit_code=${PIPESTATUS[0]}

    if (( exit_code == 0 )) && [[ -s "$checkpoint" ]] && rg -qi 'final test accuracy' "$log"; then
      printf 'OK\t%s\t%s\t%s\t%s\n' "$model" "$method" "$dataset" "$checkpoint" >> "$SUMMARY"
      return 0
    fi

    if (( attempt == 1 )) && rg -qi 'out of memory|CUDA error' "$log"; then
      attempt_batch=$(( attempt_batch / 2 ))
      (( attempt_batch < 1 )) && attempt_batch=1
      echo "GPU failure; retrying with batch=$attempt_batch"
      attempt=$((attempt + 1))
      continue
    fi

    printf 'FAIL\t%s\t%s\t%s\tlog:%s exit:%s\n' \
      "$model" "$method" "$dataset" "$log" "$exit_code" >> "$SUMMARY"
    return 1
  done
}

# model|dataset|root|batch|encoder
# Corrected non-Chest cells first, then the complete SigLIP matrix, then the
# slower full-test ChestMNIST cells.
TASKS=(
  "clip|fgvc|$DATA|8|both"
  "dino|oxford_pets|$DATA|8|vision"
  "dino|fgvc|$DATA|8|vision"
  "align|oxford_pets|$DATA|8|text"

  "siglip|caltech101|$DATA/caltech-101|2|both"
  "siglip|oxford_pets|$DATA|2|both"
  "siglip|oxford_flowers|$DATA|2|both"
  "siglip|stanford_cars|$DATA|2|both"
  "siglip|fgvc|$DATA|2|both"
  "siglip|dtd|$DATA|2|both"
  "siglip|ucf101|$DATA|2|both"
  "siglip|eurosat|$DATA|2|both"
  "siglip|food101|$DATA|2|both"
  "siglip|sun397|$DATA|2|both"
  "siglip|medmnist|$DATA/pathmnist_imagefolder|2|both"
  "siglip|imagenet_a|$DATA|2|both"
  "siglip|imagenet_r|$DATA|2|both"
  "siglip|imagenet_sketch|$DATA|2|both"
  "siglip|imagenet_v2|$DATA|2|both"

  "clip|chestmnist|$DATA/chestmnist|8|both"
  "dino|chestmnist|$DATA/chestmnist|8|vision"
  "align|chestmnist|$DATA/chestmnist|8|text"
  "siglip|chestmnist|$DATA/chestmnist|2|both"
)

echo "Finish-all queue started at $(date --iso-8601=seconds)"
for entry in "${TASKS[@]}"; do
  IFS='|' read -r model dataset root batch encoder <<< "$entry"
  for method in lora dora; do
    run_one "$model" "$dataset" "$root" "$batch" "$encoder" "$method" || true
  done
done

echo "Fine-tuning queue ended at $(date --iso-8601=seconds)"
echo "Starting gated evaluation retry."
bash "$REPO/patchsae/run_gated_matrix_retry.sh"
gated_exit=$?
echo "Finish-all pipeline ended at $(date --iso-8601=seconds), gated_exit=$gated_exit"
exit "$gated_exit"
