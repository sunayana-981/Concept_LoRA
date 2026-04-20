#!/usr/bin/env bash
# =============================================================================
# run_all_sae_inference.sh
#
# Run SAE inference and generate top-activating neuron visualizations
# for all trained SAE checkpoints.
#
# Usage:
#   chmod +x run_all_sae_inference.sh
#   ./run_all_sae_inference.sh 2>&1 | tee inference_log.txt
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/home/sunayana/Documents/Concept_LoRA/patchsae"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=0

echo "=============================================="
echo " SAE Inference — Top Activating Neuron Plots"
echo " $(date)"
echo "=============================================="
echo ""

# Run for all datasets with checkpoints
python3 run_sae_inference.py \
    --all \
    --checkpoint_root "out/checkpoints" \
    --lora_weights_root "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16" \
    --output_root "out/visualizations" \
    --model_name "openai/clip-vit-base-patch16" \
    --base_sae_path "data/sae_weight/base/out.pt" \
    --dams_css_saturation 0.5 \
    --top_k_neurons 20 \
    --top_k_images 8 \
    --max_images 5000 \
    --batch_size 32 \
    --device cuda

echo ""
echo "=============================================="
echo " Visualizations saved to: out/visualizations/"
echo "=============================================="
echo ""

# List generated files
find out/visualizations -name "*.png" -o -name "*.json" | head -50
echo ""
echo "Total files generated: $(find out/visualizations -type f | wc -l)"