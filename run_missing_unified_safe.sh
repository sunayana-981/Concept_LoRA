#!/usr/bin/env bash
set -uo pipefail

REPO=/home/sunayana/Documents/Concept_LoRA
DATA=$REPO/data
SAVE=$REPO/unified_weights
LOG_DIR=$REPO/unified_logs_missing
MANIFEST=$REPO/results/pending_unified_jobs.tsv
SUMMARY=$REPO/results/pending_unified_summary.txt
PYTHON_BIN=/home/sunayana/miniconda3/envs/patchsae/bin/python

mkdir -p "$LOG_DIR" "$SAVE" "$REPO/results"

cd "$REPO"

# Build pending matrix from full repo evidence (CSV + per-method logs + legacy checkpoints).
"$PYTHON_BIN" - <<'PY' > "$MANIFEST"
import csv
from pathlib import Path

base = Path('/home/sunayana/Documents/Concept_LoRA')
models = ['clip','dino','align']
methods = ['lora','dora']
datasets = [
    'ucf101','stanford_cars','oxford_pets','food101','oxford_flowers','fgvc','eurosat','dtd',
    'imagenet_v2','imagenet_sketch','imagenet_a','imagenet_r','caltech101','sun397','medmnist','chexpert'
]

aliases = {
    'oxford_pets': ['oxford_pets','oxford_iiit_pets'],
    'oxford_flowers': ['oxford_flowers','flowers102'],
    'fgvc': ['fgvc','fgvc_aircraft'],
    'caltech101': ['caltech101','caltech-101'],
}

def dsnames(d):
    return aliases.get(d,[d])

def mark(done, m, t, d):
    done[(m,t,d)] = True

# done matrix
keys=[(m,t,d) for m in models for t in methods for d in datasets]
done={k:False for k in keys}

# CSV evidence
csv_path = base/'results'/'accuracy_results.csv'
if csv_path.exists():
    with open(csv_path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            m = row.get('model','').strip().lower()
            t = row.get('method','').strip().lower()
            ds = row.get('dataset','').strip().lower()
            if m in models and t in methods:
                for d in datasets:
                    if ds in dsnames(d):
                        mark(done, m, t, d)

# per-method log evidence
for m in models:
    for t in methods:
        for d in datasets:
            p = base/'unified_logs'/f'{m}_{t}_{d}.log'
            if p.exists():
                txt = p.read_text(errors='ignore').lower()
                if 'final test accuracy' in txt:
                    mark(done, m, t, d)

# legacy CLIP checkpoint evidence
for d in datasets:
    for name in dsnames(d):
        if (base/'lora_weights'/'vitb16'/name/'16shots'/'seed1'/'lora_weights.pt').exists():
            mark(done, 'clip', 'lora', d)
        if (base/'dora_weights'/'vitb16'/name/'16shots'/'seed1'/'dora_weights.pt').exists():
            mark(done, 'clip', 'dora', d)

# dataset root/check-dir mapping aligned with run_all_unified.sh
def spec(d):
    if d=='ucf101': return ('data','data/UCF101')
    if d=='stanford_cars': return ('data','data/stanford_cars')
    if d=='oxford_pets': return ('data','data/oxford_pets_imagefolder')
    if d=='food101': return ('data','data/food101')
    if d=='oxford_flowers': return ('data','data/flowers102_imagefolder')
    if d=='fgvc': return ('data','data/fgvc_imagefolder')
    if d=='eurosat': return ('data','data/eurosat')
    if d=='dtd': return ('data','data/dtd')
    if d=='imagenet_v2': return ('data','data/imagenetv2')
    if d=='imagenet_sketch': return ('data','data/sketch')
    if d=='imagenet_a': return ('data','data/imagenet-a')
    if d=='imagenet_r': return ('data','data/imagenet-r')
    if d=='caltech101': return ('data/caltech-101','data/caltech-101/Caltech101')
    if d=='sun397': return ('data',str(Path.home()/'.cache/huggingface/hub/datasets--1aurent--SUN397'))
    if d=='medmnist': return ('data/pathmnist_imagefolder','data/pathmnist_imagefolder')
    if d=='chexpert': return ('data/chexpert','data/chexpert')
    raise ValueError(d)

print('model\tmethod\tdataset\troot\tcheck_dir')
for m in models:
    for t in methods:
        for d in datasets:
            if done[(m,t,d)]:
                continue
            root, chk = spec(d)
            print(f'{m}\t{t}\t{d}\t{root}\t{chk}')
PY

# common hparams
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
BATCH=16
SEED=1

ok=0
fail=0
skip=0

echo "Pending manifest: $MANIFEST"
wc -l "$MANIFEST"

echo "" > "$SUMMARY"
echo "Run started: $(date)" >> "$SUMMARY"

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r MODEL METHOD DATASET ROOT_REL CHECK_REL; do
  ROOT="$REPO/$ROOT_REL"
  CHECK="$REPO/$CHECK_REL"
  if [[ "$CHECK_REL" == /home/* ]]; then
    CHECK="$CHECK_REL"
  fi

  if [[ ! -d "$CHECK" ]]; then
    echo "SKIP $MODEL/$METHOD/$DATASET (missing dir: $CHECK)"
    echo "SKIP\t$MODEL\t$METHOD\t$DATASET\tmissing_dir" >> "$SUMMARY"
    skip=$((skip+1))
    continue
  fi

  LOG="$LOG_DIR/${MODEL}_${METHOD}_${DATASET}.log"
  FILETAG="${METHOD}_adapter_weights"
  RUN_BATCH="$BATCH"

  # OOM-prone sets: reduce batch size further.
  if [[ "$DATASET" == "imagenet_v2" || "$DATASET" == "imagenet_sketch" || "$DATASET" == "imagenet_a" || "$DATASET" == "imagenet_r" || "$DATASET" == "sun397" ]]; then
    RUN_BATCH=8
  fi

  echo "============================================================"
  echo "RUN $MODEL/$METHOD/$DATASET (batch=$RUN_BATCH)"
  echo "============================================================"

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PYTHON_BIN" unified_finetune.py \
    --model "$MODEL" \
    --method "$METHOD" \
    --dataset "$DATASET" \
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
    --filename "$FILETAG" \
    $NO_LP \
    2>&1 | tee "$LOG"

  if rg -q "final test accuracy" "$LOG"; then
    echo "OK\t$MODEL\t$METHOD\t$DATASET" >> "$SUMMARY"
    ok=$((ok+1))
  else
    echo "FAIL\t$MODEL\t$METHOD\t$DATASET" >> "$SUMMARY"
    fail=$((fail+1))
  fi

done

echo "Run ended: $(date)" >> "$SUMMARY"
echo "Summary file: $SUMMARY"
