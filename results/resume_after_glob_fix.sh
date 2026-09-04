#!/usr/bin/env bash
set -uo pipefail
REPO=/home/sunayana/Documents/Concept_LoRA
LOG=$REPO/results/resume_after_glob_fix.log

{
    echo "### 1/3: retry missing base SAE (dino_jumprelu) + confirm the other 5 are skipped"
    (cd "$REPO/patchsae" && bash run_train_base_sae_expanded.sh)

    echo "### 2/3: masked-finetune-expanded (54 runs, now unblocked)"
    (cd "$REPO/patchsae" && bash run_masked_finetune_expanded.sh)

    echo "### 3/3: retry the 2 failed MaPLe datasets (imagenet_v2, imagenet_sketch); rest will skip"
    (cd "$REPO" && bash run_all_maple.sh)

    echo "### RESUME COMPLETE"
} 2>&1 | tee "$LOG"
