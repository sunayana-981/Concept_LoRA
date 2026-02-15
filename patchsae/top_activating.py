import torch
import os
import json
import csv
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, BatchFeature
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import SAETester and Utils
from src.demo.core import SAETester
from tasks.utils import (
    get_all_classnames,
    get_max_acts_and_images,
    get_sae_and_vit,
    load_datasets,
)

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

def preprocess_image(image):
    """Preprocess image using CLIP processor"""
    inputs = processor(images=image, return_tensors="pt")
    return inputs['pixel_values'].squeeze(0)

class ImageFolderFlat(Dataset):
    def __init__(self, image_dir, preprocess_fn):
        self.paths = []
        self.preprocess_fn = preprocess_fn
        for root, dirs, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    self.paths.append(os.path.join(root, f))
        self.paths = sorted(self.paths)
        print(f"Found {len(self.paths)} images in {image_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.preprocess_fn(img), self.paths[idx]
        except Exception as e:
            print(f"Error loading {self.paths[idx]}: {e}")
            return torch.zeros(3, 224, 224), self.paths[idx]

class NpzDataset(Dataset):
    """Dataset that loads images from a .npz file (e.g., MedMNIST)."""
    def __init__(self, npz_path, preprocess_fn, split='test'):
        data = np.load(npz_path)
        # MedMNIST format: train_images, val_images, test_images
        if f'{split}_images' in data:
            self.images = data[f'{split}_images']
            self.labels = data.get(f'{split}_labels', None)
        elif 'images' in data:
            self.images = data['images']
            self.labels = data.get('labels', None)
        else:
            # Try to find any image key
            for key in data.files:
                if 'image' in key.lower():
                    self.images = data[key]
                    break
            else:
                raise ValueError(f"Cannot find image data in {npz_path}. Keys: {data.files}")
            self.labels = None
        
        self.preprocess_fn = preprocess_fn
        print(f"Loaded {len(self.images)} images from {npz_path} (split='{split}'), shape={self.images.shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_array = self.images[idx]
        # Handle grayscale (H, W) -> (H, W, 3)
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        # Handle single-channel (H, W, 1) -> (H, W, 3)
        elif img_array.ndim == 3 and img_array.shape[-1] == 1:
            img_array = np.concatenate([img_array] * 3, axis=-1)
        
        img = Image.fromarray(img_array.astype(np.uint8))
        label = int(self.labels[idx]) if self.labels is not None else -1
        # Use index as the "path" identifier
        return self.preprocess_fn(img), f"img_{idx:06d}_label{label}"

def custom_collate(batch):
    images = torch.stack([item[0] for item in batch])
    paths = [item[1] for item in batch]
    return images, paths

def convert_openai_to_hf_clip(openai_state_dict):
    hf_state_dict = OrderedDict()
    for key, value in openai_state_dict.items():
        new_key = None
        if key.startswith('visual.'):
            new_key = key.replace('visual.', 'vision_model.')
            new_key = new_key.replace('vision_model.conv1.', 'vision_model.embeddings.patch_embedding.')
            if new_key == 'vision_model.class_embedding':
                new_key = 'vision_model.embeddings.class_embedding'
            elif new_key == 'vision_model.positional_embedding':
                new_key = 'vision_model.embeddings.position_embedding.weight'
            new_key = new_key.replace('vision_model.ln_pre.', 'vision_model.pre_layrnorm.')
            new_key = new_key.replace('vision_model.ln_post.', 'vision_model.post_layernorm.')
            new_key = new_key.replace('vision_model.transformer.resblocks.', 'vision_model.encoder.layers.')
            if '.attn.in_proj_weight' in new_key:
                layer_num = new_key.split('encoder.layers.')[1].split('.')[0]
                base = f'vision_model.encoder.layers.{layer_num}.self_attn.'
                dim = value.shape[0] // 3
                hf_state_dict[base + 'q_proj.weight'] = value[:dim].clone()
                hf_state_dict[base + 'k_proj.weight'] = value[dim:2*dim].clone()
                hf_state_dict[base + 'v_proj.weight'] = value[2*dim:].clone()
                continue
            elif '.attn.in_proj_bias' in new_key:
                layer_num = new_key.split('encoder.layers.')[1].split('.')[0]
                base = f'vision_model.encoder.layers.{layer_num}.self_attn.'
                dim = value.shape[0] // 3
                hf_state_dict[base + 'q_proj.bias'] = value[:dim].clone()
                hf_state_dict[base + 'k_proj.bias'] = value[dim:2*dim].clone()
                hf_state_dict[base + 'v_proj.bias'] = value[2*dim:].clone()
                continue
            new_key = new_key.replace('.attn.out_proj.', '.self_attn.out_proj.')
            new_key = new_key.replace('.mlp.c_fc.', '.mlp.fc1.')
            new_key = new_key.replace('.mlp.c_proj.', '.mlp.fc2.')
            new_key = new_key.replace('.ln_1.', '.layer_norm1.')
            new_key = new_key.replace('.ln_2.', '.layer_norm2.')
            if new_key == 'vision_model.proj':
                new_key = 'visual_projection.weight'
                value = value.T
        elif key.startswith('transformer.') or key in ['token_embedding.weight', 'positional_embedding', 'ln_final.weight', 'ln_final.bias', 'text_projection']:
            if key == 'token_embedding.weight':
                new_key = 'text_model.embeddings.token_embedding.weight'
            elif key == 'positional_embedding':
                new_key = 'text_model.embeddings.position_embedding.weight'
            elif key.startswith('transformer.resblocks.'):
                new_key = key.replace('transformer.resblocks.', 'text_model.encoder.layers.')
                if '.attn.in_proj_weight' in new_key:
                    layer_num = new_key.split('encoder.layers.')[1].split('.')[0]
                    base = f'text_model.encoder.layers.{layer_num}.self_attn.'
                    dim = value.shape[0] // 3
                    hf_state_dict[base + 'q_proj.weight'] = value[:dim].clone()
                    hf_state_dict[base + 'k_proj.weight'] = value[dim:2*dim].clone()
                    hf_state_dict[base + 'v_proj.weight'] = value[2*dim:].clone()
                    continue
                elif '.attn.in_proj_bias' in new_key:
                    layer_num = new_key.split('encoder.layers.')[1].split('.')[0]
                    base = f'text_model.encoder.layers.{layer_num}.self_attn.'
                    dim = value.shape[0] // 3
                    hf_state_dict[base + 'q_proj.bias'] = value[:dim].clone()
                    hf_state_dict[base + 'k_proj.bias'] = value[dim:2*dim].clone()
                    hf_state_dict[base + 'v_proj.bias'] = value[2*dim:].clone()
                    continue
                new_key = new_key.replace('.attn.out_proj.', '.self_attn.out_proj.')
                new_key = new_key.replace('.mlp.c_fc.', '.mlp.fc1.')
                new_key = new_key.replace('.mlp.c_proj.', '.mlp.fc2.')
                new_key = new_key.replace('.ln_1.', '.layer_norm1.')
                new_key = new_key.replace('.ln_2.', '.layer_norm2.')
            elif key.startswith('ln_final.'):
                new_key = key.replace('ln_final.', 'text_model.final_layer_norm.')
            elif key == 'text_projection':
                new_key = 'text_projection.weight'
                value = value.T
        elif key == 'logit_scale':
            new_key = 'logit_scale'
        if new_key:
            hf_state_dict[new_key] = value
    return hf_state_dict

def get_sae_acts(sae_model, patches):
    result = sae_model(patches)
    if isinstance(result, tuple):
        if len(result) == 4:
            _, _, acts, _ = result
        elif len(result) == 3:
            _, acts, _ = result
        elif len(result) == 2:
            _, acts = result
        else:
            acts = result[0]
    else:
        acts = result
    return acts

def get_top_neurons(vit_model, sae_model, image_input, device, top_k=20, patch_idx=None):
    """
    Runs an image through the ViT and SAE to find the top-k activating neurons.
    
    Args:
        vit_model: ViT model wrapper
        sae_model: SAE model
        image_input: image tensor [1, 3, 224, 224] or [3, 224, 224]
        device: torch device
        top_k: number of top neurons to return
        patch_idx: if specified, get top neurons for this specific patch index (0-195).
                   If None, get top neurons across all patches (max over patches).
    
    Returns:
        top_neuron_indices, top_neuron_values, top_neuron_mean_values, acts
    """
    with torch.no_grad():
        if not isinstance(image_input, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(image_input)}")
        if image_input.dim() == 3:
            image_input = image_input.unsqueeze(0)
        image_input = image_input.to(device)

        image_features = vit_model.model.vision_model(pixel_values=image_input).last_hidden_state
        patches = image_features[:, 1:, :]  # Skip CLS

        acts = get_sae_acts(sae_model, patches)
        # acts shape: [1, num_patches, dict_size]

        if patch_idx is not None:
            # Get activations for a specific patch
            patch_acts = acts[0, patch_idx, :]  # [dict_size]
            top_vals, top_idxs = torch.topk(patch_acts, k=top_k)
            mean_vals = patch_acts[top_idxs]  # For a single patch, mean == value
        else:
            # Max activation per neuron across all patches
            max_per_neuron, _ = acts[0].max(dim=0)  # [dict_size]
            mean_per_neuron = acts[0].mean(dim=0)  # [dict_size]
            top_vals, top_idxs = torch.topk(max_per_neuron, k=top_k)
            mean_vals = mean_per_neuron[top_idxs]

        top_neuron_indices = top_idxs.cpu().tolist()
        top_neuron_values = top_vals.cpu().tolist()
        top_neuron_mean_values = mean_vals.cpu().tolist()

        return top_neuron_indices, top_neuron_values, top_neuron_mean_values, acts

def main():
    # Configuration
    root = "out/feature_data"
    sae_runname = "sae_base"
    vit_name = "base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae_path = "data/sae_weight/base/out.pt"
    merged_weights_path = "/home/sunayana/Documents/Concept_LoRA/clip_vitb16_eurosat_lora_merged.pt"
    image_dir = "/home/sunayana/Documents/Concept_LoRA/data/eurosat/2750"
    output_dir = "diff_neuron_plots/eurosat/"
    
    top_k = 5
    max_images = 14000  # Reduced for diagnostic
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"Using device: {device}") 
    
    datasets = load_datasets()
    

    print("Loading SAE and Base model...")
    sae, base_vit_wrapper, cfg = get_sae_and_vit(
        sae_path=sae_path,
        vit_type="base",
        device=device,
        backbone="openai/clip-vit-base-patch16",
        model_path=None,
        classnames=None,
    )
    base_vit_wrapper.model.eval()
    print(f"SAE object ID: {id(sae)}")
    print(f"Base ViT object ID: {id(base_vit_wrapper.model)}")

    print("\nLoading LoRA model...")
    _, lora_vit_wrapper, _ = get_sae_and_vit(
        sae_path=sae_path,
        vit_type="base",
        device=device,
        backbone="openai/clip-vit-base-patch16",
    )
    print(f"LoRA ViT object ID (before loading weights): {id(lora_vit_wrapper.model)}")
    
    # Check if the models are the same object
    print(f"\nAre base and LoRA the SAME object? {base_vit_wrapper.model is lora_vit_wrapper.model}")
    
    state_dict = torch.load(merged_weights_path, map_location=device)
    hf_state_dict = convert_openai_to_hf_clip(state_dict)
    res = lora_vit_wrapper.model.load_state_dict(hf_state_dict, strict=False)
    print(f"Loaded LoRA weights - Missing: {len(res.missing_keys)}, Unexpected: {len(res.unexpected_keys)}")
    lora_vit_wrapper.model.eval()
    
    # Sample a weight to verify models are different
    base_weight = base_vit_wrapper.model.vision_model.encoder.layers[0].self_attn.q_proj.weight
    lora_weight = lora_vit_wrapper.model.vision_model.encoder.layers[0].self_attn.q_proj.weight
    print(f"\nWeight comparison (layer 0 q_proj):")
    print(f"  Base mean: {base_weight.mean().item():.6f}")
    print(f"  LoRA mean: {lora_weight.mean().item():.6f}")
    print(f"  Weights are same object? {base_weight is lora_weight}")
    print(f"  Weight difference (L2): {(base_weight - lora_weight).norm().item():.6f}")

    print("\nStarting diagnostic loop...")
    
    # Detect dataset format
    if os.path.isfile(image_dir) and image_dir.endswith('.npz'):
        # .npz file (e.g., MedMNIST)
        dataset = NpzDataset(image_dir, preprocess_image, split='test')
    elif os.path.isdir(image_dir):
        # Check if directory contains .npz files
        npz_files = [f for f in os.listdir(image_dir) if f.endswith('.npz')]
        if npz_files and not any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')) 
                                  for f in os.listdir(image_dir)):
            # Directory contains .npz file(s), use the first one
            npz_path = os.path.join(image_dir, npz_files[0])
            dataset = NpzDataset(npz_path, preprocess_image, split='test')
        else:
            # Regular image folder
            dataset = ImageFolderFlat(image_dir, preprocess_image)
    else:
        raise ValueError(f"Invalid image_dir: {image_dir}. Must be a directory or .npz file.")

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=custom_collate)

    count = 0
    same_top_neuron_count = 0
    same_patch_top_neuron_count = 0
    target_patch = 88
    all_records = []

    # Count total images for tqdm
    total_images = min(len(dataset), max_images)
    pbar = tqdm(total=total_images, desc="Processing images")

    for batch_idx, (images, paths) in enumerate(dataloader):
        images = images.to(device)

        for i in range(images.shape[0]):
            img_tensor = images[i:i+1]
            img_path = paths[i]

            # Global top neurons
            base_idxs, base_vals, _, base_acts = get_top_neurons(base_vit_wrapper, sae, img_tensor, device, top_k=top_k)
            lora_idxs, lora_vals, _, lora_acts = get_top_neurons(lora_vit_wrapper, sae, img_tensor, device, top_k=top_k)

            # Patch-specific top neurons
            base_patch_idxs, base_patch_vals, _, _ = get_top_neurons(
                base_vit_wrapper, sae, img_tensor, device, top_k=top_k, patch_idx=target_patch
            )
            lora_patch_idxs, lora_patch_vals, _, _ = get_top_neurons(
                lora_vit_wrapper, sae, img_tensor, device, top_k=top_k, patch_idx=target_patch
            )

            global_same = base_idxs[0] == lora_idxs[0]
            patch_same = base_patch_idxs[0] == lora_patch_idxs[0]

            if global_same:
                same_top_neuron_count += 1
            if patch_same:
                same_patch_top_neuron_count += 1

            # Overlap stats
            global_overlap_5 = len(set(base_idxs[:5]) & set(lora_idxs[:5]))
            global_overlap_all = len(set(base_idxs) & set(lora_idxs))
            patch_overlap_5 = len(set(base_patch_idxs[:5]) & set(lora_patch_idxs[:5]))
            patch_overlap_all = len(set(base_patch_idxs) & set(lora_patch_idxs))

            base_only_global = sorted(set(base_idxs) - set(lora_idxs))
            lora_only_global = sorted(set(lora_idxs) - set(base_idxs))
            base_only_patch = sorted(set(base_patch_idxs) - set(lora_patch_idxs))
            lora_only_patch = sorted(set(lora_patch_idxs) - set(base_patch_idxs))

            # Activation diff stats at patch
            base_patch_all_acts = base_acts[0, target_patch, :].cpu()
            lora_patch_all_acts = lora_acts[0, target_patch, :].cpu()
            act_diff = base_patch_all_acts - lora_patch_all_acts
            l2_diff = act_diff.norm().item()
            max_abs_diff = act_diff.abs().max().item()
            mean_abs_diff = act_diff.abs().mean().item()

            # Top-5 most changed neurons
            top_changed = act_diff.abs().topk(5)
            most_changed = []
            for idx in range(5):
                neuron_id = top_changed.indices[idx].item()
                base_a = base_patch_all_acts[neuron_id].item()
                lora_a = lora_patch_all_acts[neuron_id].item()
                delta = top_changed.values[idx].item()
                most_changed.append({
                    'neuron': neuron_id,
                    'base_act': round(base_a, 4),
                    'lora_act': round(lora_a, 4),
                    'delta': round(delta, 4),
                })

            record = {
                'image_idx': count,
                'image_path': str(img_path),
                'image_name': os.path.basename(str(img_path)),
                'global_top1_same': global_same,
                'global_base_top1_neuron': base_idxs[0],
                'global_base_top1_act': round(base_vals[0], 4),
                'global_lora_top1_neuron': lora_idxs[0],
                'global_lora_top1_act': round(lora_vals[0], 4),
                'global_base_top5': base_idxs[:5],
                'global_base_top5_acts': [round(v, 4) for v in base_vals[:5]],
                'global_lora_top5': lora_idxs[:5],
                'global_lora_top5_acts': [round(v, 4) for v in lora_vals[:5]],
                'global_top5_overlap': global_overlap_5,
                'global_topk_overlap': global_overlap_all,
                'global_unique_to_base': base_only_global,
                'global_unique_to_lora': lora_only_global,
                f'patch{target_patch}_top1_same': patch_same,
                f'patch{target_patch}_base_top1_neuron': base_patch_idxs[0],
                f'patch{target_patch}_base_top1_act': round(base_patch_vals[0], 4),
                f'patch{target_patch}_lora_top1_neuron': lora_patch_idxs[0],
                f'patch{target_patch}_lora_top1_act': round(lora_patch_vals[0], 4),
                f'patch{target_patch}_base_top5': base_patch_idxs[:5],
                f'patch{target_patch}_base_top5_acts': [round(v, 4) for v in base_patch_vals[:5]],
                f'patch{target_patch}_lora_top5': lora_patch_idxs[:5],
                f'patch{target_patch}_lora_top5_acts': [round(v, 4) for v in lora_patch_vals[:5]],
                f'patch{target_patch}_top5_overlap': patch_overlap_5,
                f'patch{target_patch}_topk_overlap': patch_overlap_all,
                f'patch{target_patch}_unique_to_base': base_only_patch,
                f'patch{target_patch}_unique_to_lora': lora_only_patch,
                f'patch{target_patch}_l2_diff': round(l2_diff, 4),
                f'patch{target_patch}_max_abs_diff': round(max_abs_diff, 4),
                f'patch{target_patch}_mean_abs_diff': round(mean_abs_diff, 6),
                f'patch{target_patch}_most_changed_neurons': most_changed,
            }
            all_records.append(record)

            # Update progress bar with running stats
            pbar.update(1)
            pbar.set_postfix({
                'g_same': f'{same_top_neuron_count}/{count+1}',
                'p_same': f'{same_patch_top_neuron_count}/{count+1}',
            })

            count += 1
            if count >= max_images:
                break
        if count >= max_images:
            break

    pbar.close()

    # ==================== SAVE CSV FILES ====================

    # --- 1. Detailed per-image CSV ---
    csv_detail_path = os.path.join(output_dir, "neuron_comparison_detailed.csv")
    with open(csv_detail_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'image_idx', 'image_name', 'image_path',
            # Global columns
            'global_top1_same',
            'global_base_top1_neuron', 'global_base_top1_act',
            'global_lora_top1_neuron', 'global_lora_top1_act',
            'global_base_top5', 'global_base_top5_acts',
            'global_lora_top5', 'global_lora_top5_acts',
            'global_top5_overlap', f'global_top{top_k}_overlap',
            'global_unique_to_base', 'global_unique_to_lora',
            # Patch columns
            f'patch{target_patch}_top1_same',
            f'patch{target_patch}_base_top1_neuron', f'patch{target_patch}_base_top1_act',
            f'patch{target_patch}_lora_top1_neuron', f'patch{target_patch}_lora_top1_act',
            f'patch{target_patch}_base_top5', f'patch{target_patch}_base_top5_acts',
            f'patch{target_patch}_lora_top5', f'patch{target_patch}_lora_top5_acts',
            f'patch{target_patch}_top5_overlap', f'patch{target_patch}_top{top_k}_overlap',
            f'patch{target_patch}_unique_to_base', f'patch{target_patch}_unique_to_lora',
            # Activation diff stats
            f'patch{target_patch}_l2_diff', f'patch{target_patch}_max_abs_diff', f'patch{target_patch}_mean_abs_diff',
            # Most changed neurons at patch
            f'patch{target_patch}_changed1_neuron', f'patch{target_patch}_changed1_base', f'patch{target_patch}_changed1_lora', f'patch{target_patch}_changed1_delta',
            f'patch{target_patch}_changed2_neuron', f'patch{target_patch}_changed2_base', f'patch{target_patch}_changed2_lora', f'patch{target_patch}_changed2_delta',
            f'patch{target_patch}_changed3_neuron', f'patch{target_patch}_changed3_base', f'patch{target_patch}_changed3_lora', f'patch{target_patch}_changed3_delta',
            f'patch{target_patch}_changed4_neuron', f'patch{target_patch}_changed4_base', f'patch{target_patch}_changed4_lora', f'patch{target_patch}_changed4_delta',
            f'patch{target_patch}_changed5_neuron', f'patch{target_patch}_changed5_base', f'patch{target_patch}_changed5_lora', f'patch{target_patch}_changed5_delta',
        ])

        for r in all_records:
            mc = r[f'patch{target_patch}_most_changed_neurons']
            row = [
                r['image_idx'], r['image_name'], r['image_path'],
                r['global_top1_same'],
                r['global_base_top1_neuron'], r['global_base_top1_act'],
                r['global_lora_top1_neuron'], r['global_lora_top1_act'],
                str(r['global_base_top5']), str(r['global_base_top5_acts']),
                str(r['global_lora_top5']), str(r['global_lora_top5_acts']),
                r['global_top5_overlap'], r['global_topk_overlap'],
                str(r['global_unique_to_base']), str(r['global_unique_to_lora']),
                r[f'patch{target_patch}_top1_same'],
                r[f'patch{target_patch}_base_top1_neuron'], r[f'patch{target_patch}_base_top1_act'],
                r[f'patch{target_patch}_lora_top1_neuron'], r[f'patch{target_patch}_lora_top1_act'],
                str(r[f'patch{target_patch}_base_top5']), str(r[f'patch{target_patch}_base_top5_acts']),
                str(r[f'patch{target_patch}_lora_top5']), str(r[f'patch{target_patch}_lora_top5_acts']),
                r[f'patch{target_patch}_top5_overlap'], r[f'patch{target_patch}_topk_overlap'],
                str(r[f'patch{target_patch}_unique_to_base']), str(r[f'patch{target_patch}_unique_to_lora']),
                r[f'patch{target_patch}_l2_diff'], r[f'patch{target_patch}_max_abs_diff'], r[f'patch{target_patch}_mean_abs_diff'],
            ]
            # Add most changed neurons (5 entries × 4 fields)
            for ci in range(5):
                if ci < len(mc):
                    row.extend([mc[ci]['neuron'], mc[ci]['base_act'], mc[ci]['lora_act'], mc[ci]['delta']])
                else:
                    row.extend(['', '', '', ''])
            writer.writerow(row)

    print(f"Saved detailed CSV: {csv_detail_path}")

    # --- 2. Per-image patch-level rank table CSV (one row per rank per image) ---
    csv_ranks_path = os.path.join(output_dir, "neuron_ranks_per_image.csv")
    with open(csv_ranks_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'image_idx', 'image_name',
            'level', 'rank',
            'base_neuron', 'base_act',
            'lora_neuron', 'lora_act',
            'match',
            'base_neuron_in_lora_rank', 'lora_neuron_in_base_rank',
        ])
        for r in all_records:
            # Global ranks
            for rank in range(len(r['global_base_top5'])):
                b_neuron = r['global_base_top5'][rank]
                l_neuron = r['global_lora_top5'][rank]
                match = b_neuron == l_neuron
                b_in_l = r['global_lora_top5'].index(b_neuron) + 1 if b_neuron in r['global_lora_top5'] else ''
                l_in_b = r['global_base_top5'].index(l_neuron) + 1 if l_neuron in r['global_base_top5'] else ''
                writer.writerow([
                    r['image_idx'], r['image_name'],
                    'global', rank + 1,
                    b_neuron, r['global_base_top5_acts'][rank],
                    l_neuron, r['global_lora_top5_acts'][rank],
                    match, b_in_l, l_in_b,
                ])
            # Patch ranks
            for rank in range(len(r[f'patch{target_patch}_base_top5'])):
                b_neuron = r[f'patch{target_patch}_base_top5'][rank]
                l_neuron = r[f'patch{target_patch}_lora_top5'][rank]
                match = b_neuron == l_neuron
                b_in_l = r[f'patch{target_patch}_lora_top5'].index(b_neuron) + 1 if b_neuron in r[f'patch{target_patch}_lora_top5'] else ''
                l_in_b = r[f'patch{target_patch}_base_top5'].index(l_neuron) + 1 if l_neuron in r[f'patch{target_patch}_base_top5'] else ''
                writer.writerow([
                    r['image_idx'], r['image_name'],
                    f'patch{target_patch}', rank + 1,
                    b_neuron, r[f'patch{target_patch}_base_top5_acts'][rank],
                    l_neuron, r[f'patch{target_patch}_lora_top5_acts'][rank],
                    match, b_in_l, l_in_b,
                ])

    print(f"Saved rank-level CSV: {csv_ranks_path}")

    # --- 3. Summary CSV ---
    csv_summary_path = os.path.join(output_dir, "neuron_comparison_summary.csv")

    # Compute summary stats
    total = len(all_records)
    if total == 0:
        print("No images processed. Check your image_dir path and dataset format.")
        return

    global_same_count = sum(1 for r in all_records if r['global_top1_same'])
    patch_same_count = sum(1 for r in all_records if r[f'patch{target_patch}_top1_same'])

    avg_global_top5_overlap = np.mean([r['global_top5_overlap'] for r in all_records])
    avg_global_topk_overlap = np.mean([r['global_topk_overlap'] for r in all_records])
    avg_patch_top5_overlap = np.mean([r[f'patch{target_patch}_top5_overlap'] for r in all_records])
    avg_patch_topk_overlap = np.mean([r[f'patch{target_patch}_topk_overlap'] for r in all_records])
    avg_l2_diff = np.mean([r[f'patch{target_patch}_l2_diff'] for r in all_records])
    avg_max_abs_diff = np.mean([r[f'patch{target_patch}_max_abs_diff'] for r in all_records])
    avg_mean_abs_diff = np.mean([r[f'patch{target_patch}_mean_abs_diff'] for r in all_records])

    with open(csv_summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['total_images', total])
        writer.writerow(['top_k', top_k])
        writer.writerow(['target_patch', target_patch])
        writer.writerow([''])
        writer.writerow(['--- Global ---', ''])
        writer.writerow(['global_same_top1_count', global_same_count])
        writer.writerow(['global_same_top1_pct', f'{100*global_same_count/total:.1f}%'])
        writer.writerow(['global_diff_top1_count', total - global_same_count])
        writer.writerow(['global_diff_top1_pct', f'{100*(total - global_same_count)/total:.1f}%'])
        writer.writerow(['global_avg_top5_overlap', f'{avg_global_top5_overlap:.2f}/5'])
        writer.writerow([f'global_avg_top{top_k}_overlap', f'{avg_global_topk_overlap:.2f}/{top_k}'])
        writer.writerow([''])
        writer.writerow([f'--- Patch {target_patch} ---', ''])
        writer.writerow([f'patch{target_patch}_same_top1_count', patch_same_count])
        writer.writerow([f'patch{target_patch}_same_top1_pct', f'{100*patch_same_count/total:.1f}%'])
        writer.writerow([f'patch{target_patch}_diff_top1_count', total - patch_same_count])
        writer.writerow([f'patch{target_patch}_diff_top1_pct', f'{100*(total - patch_same_count)/total:.1f}%'])
        writer.writerow([f'patch{target_patch}_avg_top5_overlap', f'{avg_patch_top5_overlap:.2f}/5'])
        writer.writerow([f'patch{target_patch}_avg_top{top_k}_overlap', f'{avg_patch_topk_overlap:.2f}/{top_k}'])
        writer.writerow([''])
        writer.writerow([f'--- Patch {target_patch} Activation Diffs ---', ''])
        writer.writerow([f'patch{target_patch}_avg_l2_diff', f'{avg_l2_diff:.4f}'])
        writer.writerow([f'patch{target_patch}_avg_max_abs_diff', f'{avg_max_abs_diff:.4f}'])
        writer.writerow([f'patch{target_patch}_avg_mean_abs_diff', f'{avg_mean_abs_diff:.6f}'])

    print(f"Saved summary CSV: {csv_summary_path}")

    # --- 4. JSON with full data ---
    json_path = os.path.join(output_dir, "neuron_comparison_all.json")
    with open(json_path, 'w') as f:
        json.dump(all_records, f, indent=2)
    print(f"Saved full JSON: {json_path}")

    # --- Print final summary ---
    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"{'='*80}")
    print(f"Total images: {total}")
    print(f"Same global top-1 neuron:        {global_same_count}/{total} ({100*global_same_count/total:.1f}%)")
    print(f"Diff global top-1 neuron:        {total - global_same_count}/{total} ({100*(total - global_same_count)/total:.1f}%)")
    print(f"Same patch-{target_patch} top-1 neuron:   {patch_same_count}/{total} ({100*patch_same_count/total:.1f}%)")
    print(f"Diff patch-{target_patch} top-1 neuron:   {total - patch_same_count}/{total} ({100*(total - patch_same_count)/total:.1f}%)")
    print(f"\nAvg global top-5 overlap:        {avg_global_top5_overlap:.2f}/5")
    print(f"Avg patch-{target_patch} top-5 overlap:   {avg_patch_top5_overlap:.2f}/5")
    print(f"Avg patch-{target_patch} L2 diff:         {avg_l2_diff:.4f}")
    print(f"Avg patch-{target_patch} max abs diff:    {avg_max_abs_diff:.4f}")
    print(f"\nFiles saved to: {output_dir}/")
    print(f"  - {csv_detail_path}")
    print(f"  - {csv_ranks_path}")
    print(f"  - {csv_summary_path}")
    print(f"  - {json_path}")
    print("Done!")

if __name__ == "__main__":
    main()