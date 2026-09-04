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

# Common hyper-parameters
BACKBONE="ViT-B/16"
SHOTS=16
N_ITERS=100
LR=2e-4
RANK=4
ALPHA=1
POSITION=all
ENCODER=both
PARAMS="q k v"
NO_LP="--no_linear_probe"
SEED=1
BASE_BATCH=16
 
echo -e "status\tmodel\tmethod\tdataset_key\tdataset_label\tdetail" > "$SUMMARY"
echo "Run started: $(date)"

run_one() {
  local MODEL_KEY=$1
  local METHOD=$2
  local DATASET_LABEL=$3
  local DATASET_KEY=$4
  local ROOT=$5
  local CHECK=$6

  if [[ "$MODEL_KEY" == "siglip" ]]; then
    echo -e "SKIP\tsiglip\t$METHOD\t$DATASET_KEY\t$DATASET_LABEL\tmodel_not_supported_in_unified_finetune" >> "$SUMMARY"
    echo "SKIP siglip/$METHOD/$DATASET_LABEL (model not supported)"
    return 0
  fi

  if [[ ! -d "$CHECK" ]]; then
    echo -e "SKIP\t$MODEL_KEY\t$METHOD\t$DATASET_KEY\t$DATASET_LABEL\tmissing_data_dir:$CHECK" >> "$SUMMARY"
    echo "SKIP $MODEL_KEY/$METHOD/$DATASET_LABEL (missing dir: $CHECK)"
    return 0
  fi

  local RUN_BATCH=$BASE_BATCH
  if [[ "$DATASET_KEY" == "imagenet_v2" || "$DATASET_KEY" == "imagenet_sketch" || "$DATASET_KEY" == "imagenet_a" || "$DATASET_KEY" == "imagenet_r" || "$DATASET_KEY" == "sun397" ]]; then
    RUN_BATCH=8
  fi

  local LOG="$LOG_DIR/${MODEL_KEY}_${METHOD}_${DATASET_KEY}.log"

  echo "============================================================"
  echo "RUN $MODEL_KEY/$METHOD/$DATASET_KEY  (label: $DATASET_LABEL, batch=$RUN_BATCH)"
  echo "============================================================"

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PYTHON_BIN" unified_finetune.py \
    --model "$MODEL_KEY" \
    --method "$METHOD" \
    --dataset "$DATASET_KEY" \
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
    echo -e "OK\t$MODEL_KEY\t$METHOD\t$DATASET_KEY\t$DATASET_LABEL\tlog:$LOG" >> "$SUMMARY"
  else
    echo -e "FAIL\t$MODEL_KEY\t$METHOD\t$DATASET_KEY\t$DATASET_LABEL\tlog:$LOG" >> "$SUMMARY"
  fi
}

# 3-tuples: model_key|dataset_label|dataset_key|root|check_dir
TASKS=(
  "clip|Flowers102|oxford_flowers|$DATA|$DATA/flowers102_imagefolder"
  "clip|Stanford Cars|stanford_cars|$DATA|$DATA/stanford_cars"
  "clip|FGVC Aircraft|fgvc|$DATA|$DATA/fgvc_imagefolder"
  "clip|Food101|food101|$DATA|$DATA/food101"
  "clip|SUN397|sun397|$DATA|$HOME/.cache/huggingface/hub/datasets--1aurent--SUN397"
  "clip|ChestMNIST|chexpert|$DATA/chexpert|$DATA/chexpert"
  "clip|ImageNet-A|imagenet_a|$DATA|$DATA/imagenet-a"
  "clip|ImageNet-R|imagenet_r|$DATA|$DATA/imagenet-r"
  "clip|ImageNet-Sketch|imagenet_sketch|$DATA|$DATA/sketch"
  "clip|Corruptions|imagenet_v2|$DATA|$DATA/imagenetv2"

  "siglip|Oxford IIIT-Pets|oxford_pets|$DATA|$DATA/oxford_pets_imagefolder"
  "siglip|Flowers102|oxford_flowers|$DATA|$DATA/flowers102_imagefolder"
  "siglip|Stanford Cars|stanford_cars|$DATA|$DATA/stanford_cars"
  "siglip|Food101|food101|$DATA|$DATA/food101"
  "siglip|SUN397|sun397|$DATA|$HOME/.cache/huggingface/hub/datasets--1aurent--SUN397"
  "siglip|ChestMNIST|chexpert|$DATA/chexpert|$DATA/chexpert"
  "siglip|ImageNet-A|imagenet_a|$DATA|$DATA/imagenet-a"
  "siglip|ImageNet-R|imagenet_r|$DATA|$DATA/imagenet-r"
  "siglip|ImageNet-Sketch|imagenet_sketch|$DATA|$DATA/sketch"
  "siglip|Corruptions|imagenet_v2|$DATA|$DATA/imagenetv2"

  "dino|Caltech101|caltech101|$DATA/caltech-101|$DATA/caltech-101/Caltech101"
  "dino|Oxford IIIT-Pets|oxford_pets|$DATA|$DATA/oxford_pets_imagefolder"
  "dino|Flowers102|oxford_flowers|$DATA|$DATA/flowers102_imagefolder"
  "dino|Stanford Cars|stanford_cars|$DATA|$DATA/stanford_cars"
  "dino|FGVC Aircraft|fgvc|$DATA|$DATA/fgvc_imagefolder"
  "dino|DTD|dtd|$DATA|$DATA/dtd"
  "dino|UCF101|ucf101|$DATA|$DATA/UCF101"
  "dino|EuroSAT|eurosat|$DATA|$DATA/eurosat"
  "dino|Food101|food101|$DATA|$DATA/food101"
  "dino|SUN397|sun397|$DATA|$HOME/.cache/huggingface/hub/datasets--1aurent--SUN397"
  "dino|MedMNIST|medmnist|$DATA/pathmnist_imagefolder|$DATA/pathmnist_imagefolder"
  "dino|ChestMNIST|chexpert|$DATA/chexpert|$DATA/chexpert"
  "dino|ImageNet-A|imagenet_a|$DATA|$DATA/imagenet-a"
  "dino|ImageNet-R|imagenet_r|$DATA|$DATA/imagenet-r"
  "dino|ImageNet-Sketch|imagenet_sketch|$DATA|$DATA/sketch"
  "dino|Corruptions|imagenet_v2|$DATA|$DATA/imagenetv2"

  "align|Caltech101|caltech101|$DATA/caltech-101|$DATA/caltech-101/Caltech101"
  "align|Oxford IIIT-Pets|oxford_pets|$DATA|$DATA/oxford_pets_imagefolder"
  "align|Flowers102|oxford_flowers|$DATA|$DATA/flowers102_imagefolder"
  "align|Stanford Cars|stanford_cars|$DATA|$DATA/stanford_cars"
  "align|FGVC Aircraft|fgvc|$DATA|$DATA/fgvc_imagefolder"
  "align|DTD|dtd|$DATA|$DATA/dtd"
  "align|UCF101|ucf101|$DATA|$DATA/UCF101"
  "align|EuroSAT|eurosat|$DATA|$DATA/eurosat"
  "align|Food101|food101|$DATA|$DATA/food101"
  "align|SUN397|sun397|$DATA|$HOME/.cache/huggingface/hub/datasets--1aurent--SUN397"
  "align|MedMNIST|medmnist|$DATA/pathmnist_imagefolder|$DATA/pathmnist_imagefolder"
  "align|ChestMNIST|chexpert|$DATA/chexpert|$DATA/chexpert"
  "align|ImageNet-A|imagenet_a|$DATA|$DATA/imagenet-a"
  "align|ImageNet-R|imagenet_r|$DATA|$DATA/imagenet-r"
  "align|ImageNet-Sketch|imagenet_sketch|$DATA|$DATA/sketch"
  "align|Corruptions|imagenet_v2|$DATA|$DATA/imagenetv2"
)

for ENTRY in "${TASKS[@]}"; do
  IFS='|' read -r MODEL_KEY DATASET_LABEL DATASET_KEY ROOT CHECK <<< "$ENTRY"
  for METHOD in lora dora; do
    run_one "$MODEL_KEY" "$METHOD" "$DATASET_LABEL" "$DATASET_KEY" "$ROOT" "$CHECK"
  done
done

echo "Run ended: $(date)"
echo "Summary: $SUMMARY"
