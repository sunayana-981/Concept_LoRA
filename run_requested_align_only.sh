#!/usr/bin/env bash
set -uo pipefail

REPO=/home/sunayana/Documents/Concept_LoRA
DATA=$REPO/data
SAVE=$REPO/unified_weights
LOG_DIR=$REPO/unified_logs_requested
RESULTS=$REPO/results/accuracy_results.csv
SUMMARY=$REPO/results/requested_lora_dora_summary.tsv
PYTHON_BIN=/home/sunayana/miniconda3/envs/patchsae/bin/python

mkdir -p "$LOG_DIR" "$SAVE" "$REPO/results"
cd "$REPO"

BACKBONE="ViT-B/16"
SHOTS=16
N_ITERS=100
LR=2e-4
RANK=4
ALPHA=1
POSITION=all
ENCODER=text
PARAMS="q k v"
NO_LP="--no_linear_probe"
SEED=1
BASE_BATCH=16
FORCE_RERUN="${FORCE_RERUN:-0}"
START_DATASET="${START_DATASET:-}"

touch "$SUMMARY"

already_done() {
  local MODEL=$1
  local METHOD=$2
  local DS_KEY=$3
  if [[ ! -f "$SUMMARY" ]]; then
    return 1
  fi
  rg -q "^OK\t${MODEL}\t${METHOD}\t${DS_KEY}\t" "$SUMMARY"
}

run_one() {
  local METHOD=$1
  local LABEL=$2
  local DS_KEY=$3
  local ROOT=$4
  local CHECK=$5

  if [[ "$FORCE_RERUN" != "1" ]] && already_done "align" "$METHOD" "$DS_KEY"; then
    echo "SKIP align/$METHOD/$DS_KEY (already OK)"
    return 0
  fi

  if [[ ! -d "$CHECK" ]]; then
    echo -e "SKIP\talign\t$METHOD\t$DS_KEY\t$LABEL\tmissing_data_dir:$CHECK" >> "$SUMMARY"
    echo "SKIP align/$METHOD/$LABEL (missing dir: $CHECK)"
    return 0
  fi

  local RUN_BATCH=$BASE_BATCH
  if [[ "$DS_KEY" == "imagenet_v2" || "$DS_KEY" == "imagenet_sketch" || "$DS_KEY" == "imagenet_a" || "$DS_KEY" == "imagenet_r" || "$DS_KEY" == "sun397" ]]; then
    RUN_BATCH=8
  fi

  local LOG="$LOG_DIR/align_${METHOD}_${DS_KEY}.log"
  echo "============================================================"
  echo "RUN align/$METHOD/$DS_KEY  (label: $LABEL, batch=$RUN_BATCH)"
  echo "============================================================"

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false "$PYTHON_BIN" unified_finetune.py \
    --model align \
    --method "$METHOD" \
    --dataset "$DS_KEY" \
    --root_path "$ROOT" \
    --backbone "$BACKBONE" \
    --shots "$SHOTS" \
    --n_iters "$N_ITERS" \
    --lr "$LR" \
    --r "$RANK" \
    --alpha "$ALPHA" \
    --position "$POSITION" \
    --encoder "$ENCODER" \
    --params $PARAMS \
    --batch_size "$RUN_BATCH" \
    --seed "$SEED" \
    --save_path "$SAVE" \
    --filename "${METHOD}_adapter_weights" \
    --results_csv "$RESULTS" \
    $NO_LP \
    2>&1 | tee "$LOG"

  if rg -qi "final test accuracy" "$LOG"; then
    echo -e "OK\talign\t$METHOD\t$DS_KEY\t$LABEL\tlog:$LOG" >> "$SUMMARY"
  else
    echo -e "FAIL\talign\t$METHOD\t$DS_KEY\t$LABEL\tlog:$LOG" >> "$SUMMARY"
  fi
}

TASKS=(
  "Caltech101|caltech101|$DATA/caltech-101|$DATA/caltech-101/Caltech101"
  "Oxford IIIT-Pets|oxford_pets|$DATA|$DATA/oxford_pets_imagefolder"
  "Flowers102|oxford_flowers|$DATA|$DATA/flowers102_imagefolder"
  "Stanford Cars|stanford_cars|$DATA|$DATA/stanford_cars"
  "FGVC Aircraft|fgvc|$DATA|$DATA/fgvc_imagefolder"
  "DTD|dtd|$DATA|$DATA/dtd"
  "UCF101|ucf101|$DATA|$DATA/UCF101"
  "EuroSAT|eurosat|$DATA|$DATA/eurosat"
  "Food101|food101|$DATA|$DATA/food101"
  "SUN397|sun397|$DATA|$HOME/.cache/huggingface/hub/datasets--1aurent--SUN397"
  "MedMNIST|medmnist|$DATA/pathmnist_imagefolder|$DATA/pathmnist_imagefolder"
  "ChestMNIST|chexpert|$DATA/chexpert|$DATA/chexpert"
  "ImageNet-A|imagenet_a|$DATA|$DATA/imagenet-a"
  "ImageNet-R|imagenet_r|$DATA|$DATA/imagenet-r"
  "ImageNet-Sketch|imagenet_sketch|$DATA|$DATA/sketch"
  "Corruptions|imagenet_v2|$DATA|$DATA/imagenetv2"
)

STARTED=0
[[ -z "$START_DATASET" ]] && STARTED=1
for ENTRY in "${TASKS[@]}"; do
  IFS='|' read -r LABEL DS_KEY ROOT CHECK <<< "$ENTRY"
  if [[ "$STARTED" == "0" ]]; then
    if [[ "$DS_KEY" == "$START_DATASET" ]]; then
      STARTED=1
    else
      echo "SKIP align/*/$DS_KEY (before START_DATASET=$START_DATASET)"
      continue
    fi
  fi
  for METHOD in lora dora; do
    run_one "$METHOD" "$LABEL" "$DS_KEY" "$ROOT" "$CHECK"
  done
done

echo "ALIGN continuation run ended: $(date)"
