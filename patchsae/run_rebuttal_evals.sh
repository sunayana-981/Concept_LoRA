#!/usr/bin/env bash
# =============================================================================
# run_rebuttal_evals.sh
#
# Runs the three rebuttal eval stages built in Tasks 1/2 (tasks/eval_matrix.py,
# tasks/eval_steering.py), consuming whatever FT-SAE / FullFT-SAE checkpoints
# Tasks 4/5 (run_rebuttal_sae_training.sh, run_fullft_sae_training.sh) have
# registered in out/rebuttal/sae_registry.json so far:
#
#   1. New-domain matrix: {pets, eurosat, pathmnist} x {base, lora} x
#      {none, gsae, ftsae, fullftsae}  (fullftsae only actually resolves for
#      pathmnist -- other datasets have no fullftsae registered, so those
#      cells skip gracefully via eval_matrix.py's normal missing-checkpoint path).
#   2. Old-domain matrix: {imagenet_subset} x {base, lora(pathmnist)} x
#      {none, gsae, ftsae(pathmnist)} -- reuses pathmnist's LoRA + FT-SAE on
#      plain ImageNet images, to see whether a target-domain FT-SAE still
#      works back on the original pretraining domain.
#   3. Steering: {pathmnist, eurosat} x {gsae, ftsae}, top-k clamp ON/OFF.
#
# Each stage runs --dry_run first (foreground); the whole script aborts if a
# dry-run fails (a code/path bug, not a missing-checkpoint skip -- those are
# expected and don't fail the dry-run). Every stage passes --reuse_cache, so
# re-running this script after more SAEs finish training only (re)computes the
# newly-available cells.
#
# Usage:
#   chmod +x run_rebuttal_evals.sh
#   nohup ./run_rebuttal_evals.sh --imagenet_subset_dir /path/to/imagenet/val \
#       > out/rebuttal/logs/run_rebuttal_evals.log 2>&1 &
#
#   (Omit --imagenet_subset_dir to skip stage 2 -- stages 1 and 3 don't need it.)
# =============================================================================

set -euo pipefail

# =============================================================================
# CLI
# =============================================================================

IMAGENET_SUBSET_DIR=""
while [ $# -gt 0 ]; do
    case "$1" in
        --imagenet_subset_dir=*) IMAGENET_SUBSET_DIR="${1#*=}"; shift ;;
        --imagenet_subset_dir) IMAGENET_SUBSET_DIR="$2"; shift 2 ;;
        *) echo "[FATAL] Unknown argument: $1"; exit 1 ;;
    esac
done

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT="/home/sunayana/Documents/Concept_LoRA/patchsae"
REGISTRY_PATH="${PROJECT_ROOT}/out/rebuttal/sae_registry.json"
SAE_PATHS_JSON="${PROJECT_ROOT}/configs/rebuttal_sae_paths.json"
IMAGENET_LORA_JSON="${PROJECT_ROOT}/configs/rebuttal_lora_checkpoints_imagenet.json"
IMAGENET_SAE_JSON="${PROJECT_ROOT}/configs/rebuttal_sae_paths_imagenet.json"
LOG_ROOT="${PROJECT_ROOT}/out/rebuttal/logs"
CACHE_DIR="${PROJECT_ROOT}/out/rebuttal/cache"

cd "$PROJECT_ROOT"
mkdir -p "$LOG_ROOT" "$CACHE_DIR"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

log_header() {
    echo ""
    echo "###################################################################"
    echo "# $1"
    echo "# $(timestamp)"
    echo "###################################################################"
    echo ""
}

# =============================================================================
# REGENERATE --sae_paths FROM THE TRAINING REGISTRY (Tasks 4 & 5)
# =============================================================================

log_header "REGENERATING --sae_paths JSON FROM out/rebuttal/sae_registry.json"

if [ -f "$REGISTRY_PATH" ]; then
    python3 tasks/registry_to_sae_paths.py --registry "$REGISTRY_PATH" --out "$SAE_PATHS_JSON"
else
    echo "[WARN] ${REGISTRY_PATH} does not exist yet (Tasks 4/5 haven't registered"
    echo "       any checkpoints). Continuing with gsae-only; ftsae/fullftsae cells"
    echo "       will skip via eval_matrix.py's normal missing-checkpoint path."
    echo "{}" > "$SAE_PATHS_JSON"
fi

# Old-domain overrides: imagenet_subset reuses pathmnist's LoRA + FT-SAE.
python3 - "$REGISTRY_PATH" "$IMAGENET_LORA_JSON" "$IMAGENET_SAE_JSON" <<'PYEOF'
import json, os, sys
sys.path.insert(0, "tasks")
from registry_to_sae_paths import build_sae_paths

registry_path, lora_out, sae_out = sys.argv[1:4]

with open("configs/rebuttal_datasets.json") as f:
    dataset_registry = json.load(f)
pathmnist_lora = dataset_registry["pathmnist"]["lora_checkpoint"]
with open(lora_out, "w") as f:
    json.dump({"imagenet_subset": pathmnist_lora}, f, indent=2)
print(f"Wrote {lora_out}: imagenet_subset -> {pathmnist_lora}")

records = []
if os.path.exists(registry_path):
    with open(registry_path) as f:
        records = json.load(f)
sae_paths = build_sae_paths(records)
with open(sae_out, "w") as f:
    json.dump({"imagenet_subset": sae_paths.get("pathmnist", {})}, f, indent=2)
print(f"Wrote {sae_out}: imagenet_subset -> {sae_paths.get('pathmnist', {})}")
PYEOF

# =============================================================================
# STAGE RUNNER
# =============================================================================

run_stage() {
    local name=$1
    shift
    local dry_log="${LOG_ROOT}/${name}_dryrun.log"
    local real_log="${LOG_ROOT}/${name}.log"

    log_header "STAGE [${name}] -- DRY RUN"
    set +e
    "$@" --dry_run --reuse_cache 2>&1 | tee "$dry_log"
    local exit_code=${PIPESTATUS[0]}
    set -e
    if [ ${exit_code} -ne 0 ]; then
        echo ""
        echo "[ABORT] dry-run FAILED for stage '${name}' (exit ${exit_code}) -- see ${dry_log}"
        echo "        Not proceeding to the real run for this or later stages."
        exit 1
    fi
    echo "[OK] dry-run passed for '${name}'"

    log_header "STAGE [${name}] -- REAL RUN"
    set +e
    "$@" --reuse_cache 2>&1 | tee "$real_log"
    exit_code=${PIPESTATUS[0]}
    set -e
    if [ ${exit_code} -ne 0 ]; then
        echo "[FAILED] stage '${name}' (exit ${exit_code}) -- see ${real_log}"
        exit 1
    fi
    echo "[SUCCESS] stage '${name}' -- see ${real_log}"
}

# =============================================================================
# STAGE 1: NEW-DOMAIN MATRIX
# =============================================================================

run_stage "new_domain_matrix" python3 tasks/eval_matrix.py \
    --datasets pets eurosat pathmnist \
    --vit_types base lora \
    --sae_conditions none gsae ftsae fullftsae \
    --sae_paths "$SAE_PATHS_JSON" \
    --out_dir out/rebuttal/matrix/new_domain \
    --cache_dir "$CACHE_DIR"

# =============================================================================
# STAGE 2: OLD-DOMAIN MATRIX (needs a local ImageNet subset)
# =============================================================================

if [ -z "$IMAGENET_SUBSET_DIR" ]; then
    log_header "STAGE [old_domain_matrix] -- SKIPPED"
    echo "[SKIP] --imagenet_subset_dir not provided. Re-run with:"
    echo "  ./run_rebuttal_evals.sh --imagenet_subset_dir /path/to/imagenet/val"
    echo "  to also run the old-domain (ImageNet) matrix."
elif [ ! -d "$IMAGENET_SUBSET_DIR" ]; then
    log_header "STAGE [old_domain_matrix] -- SKIPPED"
    echo "[SKIP] --imagenet_subset_dir does not exist: ${IMAGENET_SUBSET_DIR}"
else
    run_stage "old_domain_matrix" python3 tasks/eval_matrix.py \
        --datasets imagenet_subset \
        --vit_types base lora \
        --sae_conditions none gsae ftsae \
        --lora_checkpoints "$IMAGENET_LORA_JSON" \
        --sae_paths "$IMAGENET_SAE_JSON" \
        --imagenet_subset_dir "$IMAGENET_SUBSET_DIR" \
        --out_dir out/rebuttal/matrix/old_domain \
        --cache_dir "$CACHE_DIR"
fi

# =============================================================================
# STAGE 3: STEERING
# =============================================================================

run_stage "steering" python3 tasks/eval_steering.py \
    --datasets pathmnist eurosat \
    --sae_conditions gsae ftsae \
    --sae_paths "$SAE_PATHS_JSON" \
    --out_dir out/rebuttal/steering \
    --cache_dir "$CACHE_DIR"

# =============================================================================
# SUMMARY
# =============================================================================

log_header "ALL STAGES DONE"
echo "Results:"
echo "  out/rebuttal/matrix/new_domain/results.csv"
[ -n "$IMAGENET_SUBSET_DIR" ] && echo "  out/rebuttal/matrix/old_domain/results.csv"
echo "  out/rebuttal/steering/steering_results.csv"
echo "Logs: ${LOG_ROOT}/"
