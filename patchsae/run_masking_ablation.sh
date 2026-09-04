#!/usr/bin/env bash
# Controlled publication ablation of the fraction of ImageNet-active SAE units
# protected during target-domain fine-tuning. All non-ablated settings are fixed.
set -euo pipefail

ROOT="/home/sunayana/Documents/Concept_LoRA/patchsae"
PYTHON="/home/sunayana/miniconda3/envs/dncbm310/bin/python"
BASE_SAE="${ROOT}/data/sae_weight/base/out.pt"
LORA_ROOT="/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16"
OUT="${ROOT}/out/masking_ablation"
TOKENS=100000
LAYER=-2
FRACTIONS=(0.0 0.1 0.2 0.4 0.6 0.8 1.0)
SEEDS=(1 2 3)
DATASETS=(caltech101 dtd eurosat fgvc pathmnist)

declare -A TRAIN_KEY=(
  [caltech101]=caltech101 [dtd]=dtd [eurosat]=eurosat
  [fgvc]=fgvc [pathmnist]=medmnist
)
declare -A LORA_PATH=(
  [caltech101]="${LORA_ROOT}/caltech101/16shots/seed1/lora_weights.pt"
  [dtd]="${LORA_ROOT}/dtd/16shots/seed42/lora_weights.pt"
  [eurosat]="${LORA_ROOT}/eurosat/16shots/seed1/lora_weights.pt"
  [fgvc]="${LORA_ROOT}/fgvc/16shots/seed1/lora_weights.pt"
  [pathmnist]="${LORA_ROOT}/medmnist/16shots/seed1/lora_weights.pt"
)

cd "$ROOT"
mkdir -p "$OUT/checkpoints" "$OUT/logs"
for required in "$PYTHON" "$BASE_SAE"; do
  [[ -e "$required" ]] || { echo "Missing: $required" >&2; exit 1; }
done

for dataset in "${DATASETS[@]}"; do
  [[ -f "${LORA_PATH[$dataset]}" ]] || { echo "Missing LoRA: ${LORA_PATH[$dataset]}" >&2; exit 1; }
  for seed in "${SEEDS[@]}"; do
    for frac in "${FRACTIONS[@]}"; do
      run_dir="$OUT/checkpoints/$dataset/pf${frac}/seed${seed}"
      log="$OUT/logs/${dataset}_pf${frac}_seed${seed}.log"
      existing=$(find "$run_dir" -type f -path '*/final_sparse_autoencoder_*/*.pt' -print -quit 2>/dev/null || true)
      if [[ -n "$existing" ]]; then
        echo "[SKIP] $dataset fraction=$frac seed=$seed -> $existing"
        continue
      fi
      "$PYTHON" tasks/train_sae_masked_finetune.py \
        --sae_checkpoint_path "$BASE_SAE" \
        --lora_checkpoint_path "${LORA_PATH[$dataset]}" \
        --dataset "${TRAIN_KEY[$dataset]}" --activity_dataset imagenet \
        --protect_frac "$frac" --activity_n_batches 50 \
        --block_layer "$LAYER" --total_training_tokens "$TOKENS" \
        --lr 0.0004 --l1_coefficient 0.00008 --batch_size 16 \
        --lr_warm_up_steps 500 --seed "$seed" --n_checkpoints 1 \
        --checkpoint_path "$run_dir" --device cuda \
        --run_name "mask_ablation_${dataset}_pf${frac}_seed${seed}" \
        >"$log" 2>&1
    done
  done
done

"$PYTHON" - "$OUT" <<'PY'
import glob, json, os, sys
out = sys.argv[1]
lora_root = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16"
lora = {
 "caltech101": f"{lora_root}/caltech101/16shots/seed1/lora_weights.pt",
 "dtd": f"{lora_root}/dtd/16shots/seed42/lora_weights.pt",
 "eurosat": f"{lora_root}/eurosat/16shots/seed1/lora_weights.pt",
 "fgvc": f"{lora_root}/fgvc/16shots/seed1/lora_weights.pt",
 "pathmnist": f"{lora_root}/medmnist/16shots/seed1/lora_weights.pt",
}
rows=[]
for p in sorted(glob.glob(f"{out}/checkpoints/*/pf*/seed*/**/*.pt", recursive=True)):
 if "/final_sparse_autoencoder_" not in p: continue
 parts=p.split(os.sep); dataset=parts[-6]
 rows.append({"dataset":dataset, "protect_frac":float(parts[-5][2:]),
              "seed":int(parts[-4][4:]), "checkpoint":p,
              "lora_checkpoint":lora[dataset]})
with open(f"{out}/manifest.json","w") as f: json.dump(rows,f,indent=2)
print(f"Wrote {len(rows)} runs to {out}/manifest.json")
PY

echo "Training complete. Evaluate with:"
echo "$PYTHON tasks/eval_masking_ablation.py --manifest $OUT/manifest.json --imagenet_dir /path/to/imagenet/val"
