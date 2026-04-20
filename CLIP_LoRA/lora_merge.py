import torch
import clip
from pathlib import Path
import sys

# ============================================================
# Configuration
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_VERSION = "ViT-B/16"

# Update these paths as needed
LORA_WEIGHTS_PATH = Path("/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16/eurosat/16shots/seed1/lora_weights.pt")
SAVE_PATH = Path("clip_vitb16_eurosat_lora_merged.pt")

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

def apply_lora_to_block(block, lora_dict, scale, layer_idx, encoder_type):
    """
    Merges LoRA weights into q, k, v projection layers of a specific transformer block.
    """
    # CLIP uses a combined in_proj_weight for Q, K, V
    if not hasattr(block.attn, "in_proj_weight"):
        print(f"Skipping {encoder_type} Layer {layer_idx}: No in_proj_weight found.")
        return

    # Get the base weight
    w = block.attn.in_proj_weight.data
    total_dim = w.shape[0]
    d_model = w.shape[1] 
    
    # Validation: CLIP combines Q, K, V, so dim should be 3 * d_model
    if total_dim != 3 * d_model:
        print(f"Warning: Unexpected weight shape in layer {layer_idx}: {w.shape}")

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

        # 3. Apply to Base Model
        # Map Q, K, V to specific slices of the combined matrix
        # slice_start = offset * d_model
        # slice_end = (offset + 1) * d_model
        
        if proj_name == "q_proj":
            w[:d_model, :] += delta_w
        elif proj_name == "k_proj":
            w[d_model:2*d_model, :] += delta_w
        elif proj_name == "v_proj":
            w[2*d_model:, :] += delta_w
            
    # print(f"  ✓ {encoder_type} Layer {layer_idx} merged.")

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

    # 3. Apply LoRA Updates
    print("\n=== Starting Merge Process ===")
    
    # We use torch.no_grad to save memory and ensure we modify data directly
    with torch.no_grad():
        
        # --- Text Encoder (Layers 0-11) ---
        print("Merging Text Encoder...")
        for i in range(12):
            layer_key = f"layer_{i}"
            if layer_key in layers:
                apply_lora_to_block(
                    model.transformer.resblocks[i], 
                    layers[layer_key], 
                    scale, 
                    layer_idx=i, 
                    encoder_type="Text"
                )
            else:
                print(f"Warning: {layer_key} missing from LoRA weights")

        # --- Vision Encoder (Layers 12-23) ---
        # Note: Concept LoRA typically maps indices 12-23 to Vision 0-11
        print("Merging Vision Encoder...")
        for i in range(12, 24):
            layer_key = f"layer_{i}"
            vision_layer_idx = i - 12
            
            if layer_key in layers:
                apply_lora_to_block(
                    model.visual.transformer.resblocks[vision_layer_idx], 
                    layers[layer_key], 
                    scale, 
                    layer_idx=vision_layer_idx, 
                    encoder_type="Vision"
                )
            else:
                print(f"Warning: {layer_key} missing from LoRA weights")

    # 4. Verify Merge
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
        
        # 5. Save Model
        print(f"\nSaving merged model to {SAVE_PATH}...")
        torch.save(model.state_dict(), SAVE_PATH)
        print("Done.")
    else:
        print("✗ FAILURE: No weight changes detected. Check LoRA keys or scale.")