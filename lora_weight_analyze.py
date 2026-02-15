import torch
import clip
from pathlib import Path
import sys
import numpy as np

# ============================================================
# Configuration
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_VERSION = "ViT-B/16"

# Update these paths as needed
LORA_WEIGHTS_PATH = Path("/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16/medmnist/16shots/seed1/lora_weights.pt")
SAVE_PATH = Path("clip_vitb16_medmnist_lora_merged.pt")

# ============================================================
# Helper Functions
# ============================================================

def inspect_lora_structure(layers):
    """Prints debug info about the LoRA weight structure."""
    print("\n=== Checking LoRA structure ===")
    if not layers:
        print("Error: LoRA weights dictionary is empty.")
        return

    # Check the first available layer to detect structure
    first_key = list(layers.keys())[0]
    sample_layer = layers[first_key]
    
    print(f"Sample key: {first_key}")
    print(f"Sample type: {type(sample_layer)}")

    # Check if nested (dict of dicts) or flat
    if isinstance(sample_layer, dict) and "q_proj" in sample_layer and isinstance(sample_layer["q_proj"], dict):
        print("Structure detected: Nested Dictionary (layer -> proj -> weights)")
        print(f"Keys in q_proj: {sample_layer['q_proj'].keys()}")
    else:
        print("Structure detected: Flat Dictionary or different nesting")
        # Print first few keys to help debug
        for k in list(sample_layer.keys())[:5]:
            val = sample_layer[k]
            info = val.shape if torch.is_tensor(val) else type(val)
            print(f"  {k}: {info}")

def apply_lora_to_block(block, lora_dict, scale, layer_idx, encoder_type, weight_changes):
    """
    Merges LoRA weights into q, k, v projection layers of a specific transformer block.
    Returns statistics about weight changes.
    """
    # CLIP uses a combined in_proj_weight for Q, K, V
    if not hasattr(block.attn, "in_proj_weight"):
        print(f"Skipping {encoder_type} Layer {layer_idx}: No in_proj_weight found.")
        return

    # Get the base weight (make a copy to calculate changes)
    w = block.attn.in_proj_weight.data
    w_original = w.clone()  # Keep a copy for comparison
    
    total_dim = w.shape[0]
    d_model = w.shape[1] 
    
    # Validation: CLIP combines Q, K, V, so dim should be 3 * d_model
    if total_dim != 3 * d_model:
        print(f"Warning: Unexpected weight shape in layer {layer_idx}: {w.shape}")

    layer_deltas = []  # Store delta_w for each projection in this layer
    
    for proj_name, offset in zip(["q_proj", "k_proj", "v_proj"], [0, 1, 2]):
        
        # 1. Extract A and B matrices
        A, B = None, None
        
        # Scenario A: Nested structure
        if proj_name in lora_dict and isinstance(lora_dict[proj_name], dict):
            try:
                A = lora_dict[proj_name]["w_lora_A"]
                B = lora_dict[proj_name]["w_lora_B"]
            except KeyError:
                pass
        
        # Scenario B: Flat structure
        if A is None:
            try:
                A = lora_dict[f"{proj_name}.w_lora_A"]
                B = lora_dict[f"{proj_name}.w_lora_B"]
            except KeyError:
                # This projection might not have trained LoRA weights
                continue

        # 2. Compute Delta
        # Move to same device as model for calculation
        A = A.to(DEVICE)
        B = B.to(DEVICE)
        
        # LoRA update: dW = scale * B @ A
        # Shape check: B=(d_out, r), A=(r, d_in) -> B@A=(d_out, d_in)
        delta_w = scale * (B @ A)
        layer_deltas.append(delta_w)

        # 3. Apply to Base Model
        # Map Q, K, V to specific slices of the combined matrix
        if proj_name == "q_proj":
            w[:d_model, :] += delta_w
        elif proj_name == "k_proj":
            w[d_model:2*d_model, :] += delta_w
        elif proj_name == "v_proj":
            w[2*d_model:, :] += delta_w
    
    # 4. Calculate weight change metrics for this layer
    if layer_deltas:
        # Calculate change for the entire layer
        w_new = block.attn.in_proj_weight.data
        delta_w_total = w_new - w_original
        
        # Calculate ∆W/W ratio
        # Use Frobenius norm for overall magnitude
        delta_norm = torch.norm(delta_w_total, p='fro').item()
        w_norm = torch.norm(w_original, p='fro').item()
        
        if w_norm > 0:
            delta_ratio = delta_norm / w_norm
        else:
            delta_ratio = 0.0
        
        # Also calculate element-wise statistics
        abs_delta = torch.abs(delta_w_total)
        abs_w = torch.abs(w_original)
        
        # Avoid division by zero
        elementwise_ratio = torch.where(
            abs_w > 1e-10,
            abs_delta / abs_w,
            torch.zeros_like(abs_w)
        )
        
        mean_elementwise_ratio = elementwise_ratio.mean().item()
        max_elementwise_ratio = elementwise_ratio.max().item()
        
        weight_changes[f"{encoder_type}_Layer_{layer_idx}"] = {
            'delta_norm': delta_norm,
            'w_norm': w_norm,
            'delta_ratio_frobenius': delta_ratio,
            'mean_elementwise_ratio': mean_elementwise_ratio,
            'max_elementwise_ratio': max_elementwise_ratio
        }
        
        print(f"  ✓ {encoder_type} Layer {layer_idx}: ∆W/W = {delta_ratio:.6f} (Frobenius), "
              f"Mean elem = {mean_elementwise_ratio:.6f}")

def print_weight_change_summary(weight_changes):
    """Print a detailed summary of weight changes across all layers."""
    print("\n" + "="*80)
    print("WEIGHT CHANGE ANALYSIS: ∆W/W per Layer")
    print("="*80)
    
    text_layers = []
    vision_layers = []
    
    for layer_name, stats in sorted(weight_changes.items()):
        if 'Text' in layer_name:
            text_layers.append(stats)
        else:
            vision_layers.append(stats)
        
        print(f"\n{layer_name}:")
        print(f"  ∆W (Frobenius norm):     {stats['delta_norm']:.6f}")
        print(f"  W (Frobenius norm):      {stats['w_norm']:.6f}")
        print(f"  ∆W/W (Frobenius):        {stats['delta_ratio_frobenius']:.6f}")
        print(f"  ∆W/W (Mean element-wise): {stats['mean_elementwise_ratio']:.6f}")
        print(f"  ∆W/W (Max element-wise):  {stats['max_elementwise_ratio']:.6f}")
    
    # Calculate averages
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    if text_layers:
        avg_text_frobenius = np.mean([l['delta_ratio_frobenius'] for l in text_layers])
        avg_text_elementwise = np.mean([l['mean_elementwise_ratio'] for l in text_layers])
        print(f"\nText Encoder (12 layers):")
        print(f"  Average ∆W/W (Frobenius):        {avg_text_frobenius:.6f}")
        print(f"  Average ∆W/W (Element-wise):     {avg_text_elementwise:.6f}")
    
    if vision_layers:
        avg_vision_frobenius = np.mean([l['delta_ratio_frobenius'] for l in vision_layers])
        avg_vision_elementwise = np.mean([l['mean_elementwise_ratio'] for l in vision_layers])
        print(f"\nVision Encoder (12 layers):")
        print(f"  Average ∆W/W (Frobenius):        {avg_vision_frobenius:.6f}")
        print(f"  Average ∆W/W (Element-wise):     {avg_vision_elementwise:.6f}")
    
    if text_layers and vision_layers:
        all_layers = text_layers + vision_layers
        overall_avg_frobenius = np.mean([l['delta_ratio_frobenius'] for l in all_layers])
        overall_avg_elementwise = np.mean([l['mean_elementwise_ratio'] for l in all_layers])
        print(f"\nOverall (24 layers):")
        print(f"  Average ∆W/W (Frobenius):        {overall_avg_frobenius:.6f}")
        print(f"  Average ∆W/W (Element-wise):     {overall_avg_elementwise:.6f}")
    
    print("="*80)

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    
    # 1. Load Resources
    print(f"Loading base CLIP model: {CLIP_VERSION}...")
    model, _ = clip.load(CLIP_VERSION, device=DEVICE)
    
    if not LORA_WEIGHTS_PATH.exists():
        print(f"Error: LoRA file not found at {LORA_WEIGHTS_PATH}")
        sys.exit(1)

    print(f"Loading LoRA weights from {LORA_WEIGHTS_PATH.name}...")
    lora_state = torch.load(LORA_WEIGHTS_PATH, map_location=DEVICE)
    
    layers = lora_state["weights"]
    meta = lora_state["metadata"]

    # Calculate Scale
    r = meta["r"]
    alpha = meta["alpha"]
    scale = alpha / r
    print(f"LoRA Params: rank={r}, alpha={alpha}, scale={scale}")
    print(f"Total LoRA layers found: {len(layers)}")

    # 2. Inspect Structure (Debug)
    inspect_lora_structure(layers)

    # 3. Apply LoRA Updates and Track Changes
    print("\n=== Starting Merge Process ===")
    
    # Dictionary to store weight change statistics
    weight_changes = {}
    
    # We use torch.no_grad to save memory and ensure we modify data directly
    with torch.no_grad():
        
        # --- Text Encoder (Layers 0-11) ---
        print("\nMerging Text Encoder...")
        for i in range(12):
            layer_key = f"layer_{i}"
            if layer_key in layers:
                apply_lora_to_block(
                    model.transformer.resblocks[i], 
                    layers[layer_key], 
                    scale, 
                    layer_idx=i, 
                    encoder_type="Text",
                    weight_changes=weight_changes
                )
            else:
                print(f"Warning: {layer_key} missing from LoRA weights")

        # --- Vision Encoder (Layers 12-23) ---
        # Note: Concept LoRA typically maps indices 12-23 to Vision 0-11
        print("\nMerging Vision Encoder...")
        for i in range(12, 24):
            layer_key = f"layer_{i}"
            vision_layer_idx = i - 12
            
            if layer_key in layers:
                apply_lora_to_block(
                    model.visual.transformer.resblocks[vision_layer_idx], 
                    layers[layer_key], 
                    scale, 
                    layer_idx=vision_layer_idx, 
                    encoder_type="Vision",
                    weight_changes=weight_changes
                )
            else:
                print(f"Warning: {layer_key} missing from LoRA weights")

    # 4. Print Weight Change Summary
    print_weight_change_summary(weight_changes)

    # 5. Verify Merge
    print("\n=== Verifying Merge Integrity ===")
    # Load a clean model to compare against
    clean_model, _ = clip.load(CLIP_VERSION, device=DEVICE)
    
    # Compare Text Encoder Layer 0
    w_merged_text = model.transformer.resblocks[0].attn.in_proj_weight
    w_clean_text = clean_model.transformer.resblocks[0].attn.in_proj_weight
    text_diff = (w_merged_text - w_clean_text).abs().max().item()
    
    # Compare Vision Encoder Layer 0
    w_merged_vis = model.visual.transformer.resblocks[0].attn.in_proj_weight
    w_clean_vis = clean_model.visual.transformer.resblocks[0].attn.in_proj_weight
    vis_diff = (w_merged_vis - w_clean_vis).abs().max().item()

    print(f"Max difference in Text Encoder Layer 0:   {text_diff:.8f}")
    print(f"Max difference in Vision Encoder Layer 0: {vis_diff:.8f}")

    if text_diff > 0 or vis_diff > 0:
        print("✓ SUCCESS: Weights have been modified.")
        
        # 6. Save Model
        print(f"\nSaving merged model to {SAVE_PATH}...")
        torch.save(model.state_dict(), SAVE_PATH)
        print("Done.")
    else:
        print("✗ FAILURE: No weight changes detected. Check LoRA keys or scale.")