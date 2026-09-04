#!/usr/bin/env bash
# Run the currently runnable missing LoRA/DoRA checkpoints sequentially.
# Known-blocked jobs (unsupported SigLIP or incomplete/missing datasets) are
# intentionally excluded; see results/remaining_runnable_status.tsv.

set -uo pipefail

REPO=/home/sunayana/Documents/Concept_LoRA
DATA=$REPO/data
SAVE=$REPO/unified_weights
LOG_DIR=$REPO/unified_logs_remaining
STATUS=$REPO/results/remaining_runnable_status.tsv
PYTHON_BIN=/home/sunayana/miniconda3/envs/patchsae/bin/python
MIN_FREE_MIB=18000

mkdir -p "$LOG_DIR" "$SAVE" "$REPO/results"
cd "$REPO"

if [[ ! -f "$STATUS" ]]; then
  printf 'timestamp\tstatus\tmodel\tmethod\tdataset\tdetail\n' > "$STATUS"
fi

wait_for_gpu() {
  local free_mib
  while true; do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1)
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
      echo "GPU ready: ${free_mib} MiB free."
      return 0
    fi
    echo "$(date '+%F %T') waiting for GPU: ${free_mib:-unknown} MiB free; need ${MIN_FREE_MIB} MiB."
    sleep 60
  done
}

run_one() {
  local method=$1
  local dataset=$2
  local root=$3
  local checkpoint="$SAVE/align-base/$dataset/16shots/seed1/${method}_adapter_weights.pt"
  local log="$LOG_DIR/align_${method}_${dataset}.log"

  if [[ -f "$checkpoint" ]]; then
    echo "SKIP align/$method/$dataset: checkpoint already exists."
    return 0
  fi

  wait_for_gpu
  echo "============================================================"
  echo "RUN align/$method/$dataset"
  echo "============================================================"
  printf '%s\tSTART\talign\t%s\t%s\tlog:%s\n' "$(date --iso-8601=seconds)" "$method" "$dataset" "$log" >> "$STATUS"

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false \
    "$PYTHON_BIN" unified_finetune.py \
      --model align \
      --method "$method" \
      --dataset "$dataset" \
      --root_path "$root" \
      --backbone ViT-B/16 \
      --shots 16 \
      --n_iters 100 \
      --lr 2e-4 \
      --r 4 \
      --alpha 1 \
      --position all \
      --encoder both \
      --params q k v \
      --batch_size 8 \
      --seed 1 \
      --save_path "$SAVE" \
      --filename "${method}_adapter_weights" \
      --results_csv "$REPO/results/accuracy_results.csv" \
      --no_linear_probe \
      2>&1 | tee "$log"

  if [[ -f "$checkpoint" ]] && rg -qi 'final test accuracy' "$log"; then
    printf '%s\tOK\talign\t%s\t%s\tcheckpoint:%s\n' "$(date --iso-8601=seconds)" "$method" "$dataset" "$checkpoint" >> "$STATUS"
  else
    printf '%s\tFAIL\talign\t%s\t%s\tlog:%s\n' "$(date --iso-8601=seconds)" "$method" "$dataset" "$log" >> "$STATUS"
  fi
}

# SUN397 DoRA was interrupted during evaluation; its checkpoint is absent.
run_one dora sun397 "$DATA"

for method in lora dora; do
  run_one "$method" medmnist "$DATA/pathmnist_imagefolder"
done

for dataset in imagenet_a imagenet_r imagenet_sketch imagenet_v2; do
  for method in lora dora; do
    run_one "$method" "$dataset" "$DATA"
  done
done

echo "Runnable remaining queue finished at $(date)."
