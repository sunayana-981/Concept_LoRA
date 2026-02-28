#!/usr/bin/env bash
# =============================================================================
# run_lora_vs_maple_comparison.sh
#
# Minimal comparison: LoRA-finetuned SAE vs MaPLe-finetuned SAE
#
# Phases:
#   Phase 1: Train masked SAE on MaPLe activations
#   Phase 2: Train masked SAE on LoRA activations  (same hyperparams)
#   Phase 3: Evaluate both checkpoints side by side
#   Phase 4: Print side-by-side summary table
#
# Usage:
#   chmod +x run_lora_vs_maple_comparison.sh
#   ./run_lora_vs_maple_comparison.sh 2>&1 | tee comparison_log.txt
#
# To skip training and just re-evaluate existing checkpoints:
#   SKIP_TRAIN=1 ./run_lora_vs_maple_comparison.sh
# =============================================================================

set -euo pipefail

# =============================================================================
# PATHS — EDIT THESE
# =============================================================================

PROJECT_ROOT="/home/sunayana/Documents/Concept_LoRA/patchsae"
BASE_SAE="${PROJECT_ROOT}/data/sae_weight/base/out.pt"

# LoRA
LORA_CHECKPOINT="/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16/medmnist/16shots/seed1/lora_weights.pt"
LORA_CKPT_DIR="${PROJECT_ROOT}/out/checkpoints/masked_finetune_lora"

# MaPLe
MAPLE_MODEL="/home/sunayana/Documents/model.pth.tar-5"
MAPLE_CONFIG="${PROJECT_ROOT}/configs/models/maple/vit_b16_c2_ep5_batch4_2ctx.yaml"
MAPLE_CKPT_DIR="${PROJECT_ROOT}/out/checkpoints/masked_finetune_maple"

# Eval output
EVAL_DIR="${PROJECT_ROOT}/out/eval_comparison"

# =============================================================================
# SHARED HYPERPARAMETERS (identical for both runs)
# =============================================================================

DATASET="medmnist"
PROTECT_FRAC=0.2
ACTIVITY_N_BATCHES=50
L1_COEFF=0.00008
BATCH_SIZE=16
TOTAL_TOKENS=1000000
BLOCK_LAYER=-2
SEED=42
DEVICE="cuda"

# =============================================================================
# SKIP_TRAIN mode
# =============================================================================
SKIP_TRAIN="${SKIP_TRAIN:-0}"

cd "${PROJECT_ROOT}"
mkdir -p "${EVAL_DIR}"

# =============================================================================
# PHASE 1: TRAIN — MaPLe
# =============================================================================
if [ "${SKIP_TRAIN}" = "0" ]; then

echo ""
echo "================================================================"
echo "  PHASE 1: MASKED SAE FINE-TUNE — MaPLe"
echo "================================================================"
echo ""

mkdir -p "${MAPLE_CKPT_DIR}"

python3 tasks/train_sae_masked_finetune.py \
    --sae_checkpoint_path "${BASE_SAE}" \
    --vit_type maple \
    --model_path "${MAPLE_MODEL}" \
    --config_path "${MAPLE_CONFIG}" \
    --block_layer -2 \
    --dataset "${DATASET}" \
    --protect_frac 0.8 \
    --activity_n_batches ${ACTIVITY_N_BATCHES} \
    --l1_coefficient ${L1_COEFF} \
    --batch_size ${BATCH_SIZE} \
    --total_training_tokens 10000 \
    --checkpoint_path "${MAPLE_CKPT_DIR}" \
    --seed ${SEED} \
    --device ${DEVICE} \
    --run_name "maple_masked_sae"

# =============================================================================
# PHASE 2: TRAIN — LoRA
# =============================================================================

echo ""
echo "================================================================"
echo "  PHASE 2: MASKED SAE FINE-TUNE — LoRA"
echo "================================================================"
echo ""

mkdir -p "${LORA_CKPT_DIR}"

python3 tasks/train_sae_masked_finetune.py \
    --sae_checkpoint_path "${BASE_SAE}" \
    --vit_type base \
    --lora_checkpoint_path "${LORA_CHECKPOINT}" \
    --block_layer -2 \
    --dataset "${DATASET}" \
    --protect_frac 0.8 \
    --activity_n_batches ${ACTIVITY_N_BATCHES} \
    --l1_coefficient ${L1_COEFF} \
    --batch_size ${BATCH_SIZE} \
    --total_training_tokens 10000 \
    --checkpoint_path "${LORA_CKPT_DIR}" \
    --seed ${SEED} \
    --device ${DEVICE} \
    --run_name "lora_masked_sae"

fi  # end SKIP_TRAIN

# =============================================================================
# PHASE 3: EVALUATE BOTH
# =============================================================================

echo ""
echo "================================================================"
echo "  PHASE 3: EVALUATION"
echo "================================================================"
echo ""

# --- Find the latest checkpoint in each directory ---
find_latest_ckpt() {
    local dir="$1"
    local ckpt=""
    ckpt=$(find "${dir}" -name "final_*.pt" -type f 2>/dev/null | sort | tail -1)
    if [ -z "${ckpt}" ]; then
        ckpt=$(find "${dir}" -name "*.pt" -type f 2>/dev/null | sort | tail -1)
    fi
    echo "${ckpt}"
}

MAPLE_SAE_CKPT=$(find_latest_ckpt "${MAPLE_CKPT_DIR}")
LORA_SAE_CKPT=$(find_latest_ckpt "${LORA_CKPT_DIR}")

if [ -z "${MAPLE_SAE_CKPT}" ]; then
    echo "[ERROR] No MaPLe SAE checkpoint found in ${MAPLE_CKPT_DIR}"
    echo "        Run training first (without SKIP_TRAIN=1)"
    exit 1
fi
if [ -z "${LORA_SAE_CKPT}" ]; then
    echo "[ERROR] No LoRA SAE checkpoint found in ${LORA_CKPT_DIR}"
    echo "        Run training first (without SKIP_TRAIN=1)"
    exit 1
fi

echo "  MaPLe SAE checkpoint: ${MAPLE_SAE_CKPT}"
echo "  LoRA SAE checkpoint:  ${LORA_SAE_CKPT}"
echo ""

# --- Eval MaPLe-finetuned SAE ---
echo "── Evaluating MaPLe-finetuned SAE ──"
python3 eval_masked_sae.py \
    --masked_sae_path "${MAPLE_SAE_CKPT}" \
    --base_sae_path "${BASE_SAE}" \
    --lora_checkpoint "${LORA_CHECKPOINT}" \
    --maple_model_path "${MAPLE_MODEL}" \
    --maple_config_path "${MAPLE_CONFIG}" \
    --save_dir "${EVAL_DIR}/maple_sae" \
    --skip_linear_probe  \
    --batch_size 64

# --- Eval LoRA-finetuned SAE ---
echo ""
echo "── Evaluating LoRA-finetuned SAE ──"
python3 eval_masked_sae.py \
    --masked_sae_path "${LORA_SAE_CKPT}" \
    --base_sae_path "${BASE_SAE}" \
    --lora_checkpoint "${LORA_CHECKPOINT}" \
    --maple_model_path "${MAPLE_MODEL}" \
    --maple_config_path "${MAPLE_CONFIG}" \
    --save_dir "${EVAL_DIR}/lora_sae" \
    --skip_linear_probe \
    --batch_size 64

# =============================================================================
# PHASE 4: SIDE-BY-SIDE SUMMARY
# =============================================================================

echo ""
echo "================================================================"
echo "  SIDE-BY-SIDE COMPARISON"
echo "================================================================"
echo ""

# Pass EVAL_DIR as argument to avoid bash/python quoting issues
python3 - "${EVAL_DIR}" << 'PYEOF'
import json
import sys
import os

eval_dir = sys.argv[1]

maple_path = os.path.join(eval_dir, "maple_sae", "eval_masked_sae_results.json")
lora_path  = os.path.join(eval_dir, "lora_sae",  "eval_masked_sae_results.json")

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  [WARN] Not found: {path}")
        return {}

maple = load_json(maple_path)
lora  = load_json(lora_path)

if not maple and not lora:
    print("  No results found. Check eval logs above.")
    sys.exit(0)

SEP = "-" * 80

def header(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def row(label, m_val, l_val):
    if m_val is not None and l_val is not None:
        delta = l_val - m_val
        print(f"  {label:<42s}  {m_val:>9.2f}%  {l_val:>9.2f}%  {delta:>+7.2f}%")
    elif m_val is not None:
        print(f"  {label:<42s}  {m_val:>9.2f}%  {'n/a':>10s}  {'':>8s}")
    elif l_val is not None:
        print(f"  {label:<42s}  {'n/a':>10s}  {l_val:>9.2f}%  {'':>8s}")

# ---------- Accuracy ----------
header("ACCURACY COMPARISON")
col_hdr = f"  {'Metric':<42s}  {'MaPLe SAE':>10s}  {'LoRA SAE':>10s}  {'Delta':>8s}"
print(col_hdr)
print(f"  {'-'*42}  {'-'*10}  {'-'*10}  {'-'*8}")

# MedMNIST (non-MaPLe sections)
mm = maple.get("medmnist_accuracy", {})
lm = lora.get("medmnist_accuracy", {})
row("MedMNIST: LoRA CLIP + masked SAE",     mm.get("lora_masked_sae"),  lm.get("lora_masked_sae"))
row("MedMNIST: LoRA CLIP + base SAE",       mm.get("lora_base_sae"),    lm.get("lora_base_sae"))
row("MedMNIST: LoRA CLIP, no SAE",          mm.get("lora_no_sae"),      lm.get("lora_no_sae"))
row("MedMNIST: Base CLIP + masked SAE",     mm.get("base_masked_sae"),  lm.get("base_masked_sae"))

# MaPLe-specific
mmp = maple.get("medmnist_accuracy_maple", {})
lmp = lora.get("medmnist_accuracy_maple", {})
row("MedMNIST: MaPLe CLIP + masked SAE",    mmp.get("maple_masked_sae"), lmp.get("maple_masked_sae"))
row("MedMNIST: MaPLe CLIP + base SAE",      mmp.get("maple_base_sae"),   lmp.get("maple_base_sae"))
row("MedMNIST: MaPLe CLIP, no SAE",         mmp.get("maple_no_sae"),     lmp.get("maple_no_sae"))

# ImageNet (if available)
mi = maple.get("imagenet_accuracy", {})
li = lora.get("imagenet_accuracy", {})
if mi or li:
    print()
    row("ImageNet: LoRA CLIP + masked SAE",  mi.get("lora_masked_sae"),  li.get("lora_masked_sae"))
    row("ImageNet: Base CLIP + masked SAE",  mi.get("base_masked_sae"),  li.get("base_masked_sae"))

# ---------- Reconstruction ----------
mr = maple.get("reconstruction", {})
lr = lora.get("reconstruction", {})
if mr or lr:
    header("RECONSTRUCTION QUALITY")
    print(f"  {'Config':<35s}  {'MaPLe SAE':>10s}  {'LoRA SAE':>10s}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*10}")
    for key in ["masked_sae_medmnist", "base_sae_medmnist",
                "masked_sae_imagenet", "base_sae_imagenet"]:
        m_r = mr.get(key, {})
        l_r = lr.get(key, {})
        if not m_r and not l_r:
            continue
        m_cos = f"{m_r['cos_sim']:.4f}" if "cos_sim" in m_r else "n/a"
        l_cos = f"{l_r['cos_sim']:.4f}" if "cos_sim" in l_r else "n/a"
        print(f"  {key + ' cos_sim':<35s}  {m_cos:>10s}  {l_cos:>10s}")
        m_mse = f"{m_r['mse']:.6f}" if "mse" in m_r else "n/a"
        l_mse = f"{l_r['mse']:.6f}" if "mse" in l_r else "n/a"
        print(f"  {key + ' mse':<35s}  {m_mse:>10s}  {l_mse:>10s}")

# ---------- Feature Analysis ----------
fa_m = maple.get("feature_analysis", {})
fa_l = lora.get("feature_analysis", {})
if fa_m or fa_l:
    header("FEATURE ANALYSIS (masked SAE on MedMNIST, Base HF model)")
    print(f"  {'Group':<12s}  {'Metric':<15s}  {'MaPLe SAE':>10s}  {'LoRA SAE':>10s}")
    print(f"  {'-'*12}  {'-'*15}  {'-'*10}  {'-'*10}")
    for analysis_key in ["masked_sae_on_medmnist"]:
        m_fa = fa_m.get(analysis_key, {})
        l_fa = fa_l.get(analysis_key, {})
        for group in ["protected", "free", "all"]:
            m_g = m_fa.get(group, {})
            l_g = l_fa.get(group, {})
            if not m_g and not l_g:
                continue
            for metric in ["dead_fraction", "fire_rate_mean", "mean_activity"]:
                m_v = m_g.get(metric)
                l_v = l_g.get(metric)
                m_s = f"{m_v:.6f}" if m_v is not None else "n/a"
                l_s = f"{l_v:.6f}" if l_v is not None else "n/a"
                print(f"  {group:<12s}  {metric:<15s}  {m_s:>10s}  {l_s:>10s}")

print(f"\n{SEP}")
print(f"  Full results:")
print(f"    MaPLe SAE: {maple_path}")
print(f"    LoRA SAE:  {lora_path}")
print(SEP)
PYEOF

echo ""
echo "Done."