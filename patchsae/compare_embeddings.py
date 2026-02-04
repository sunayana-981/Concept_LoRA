"""
Compare SAE and CLIP embeddings between base CLIP and LoRA fine-tuned model.
Processes entire datasets and saves summary statistics.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from transformers import CLIPProcessor
import torchvision.transforms as transforms
from torchvision.datasets import UCF101
from PIL import Image

from tasks.utils import (
    get_sae_and_vit,
    load_datasets,
)


def convert_openai_to_hf_clip(openai_state_dict):
    """
    Convert full OpenAI CLIP state dict to HuggingFace CLIP format.
    Handles both vision and text models.
    """
    hf_state_dict = OrderedDict()
    
    for key, value in openai_state_dict.items():
        new_key = None
        
        # Handle vision model keys
        if key.startswith('visual.'):
            new_key = key.replace('visual.', 'vision_model.')
            
            # Patch embedding
            new_key = new_key.replace('vision_model.conv1.', 'vision_model.embeddings.patch_embedding.')
            
            # Class and position embeddings
            if new_key == 'vision_model.class_embedding':
                new_key = 'vision_model.embeddings.class_embedding'
            elif new_key == 'vision_model.positional_embedding':
                new_key = 'vision_model.embeddings.position_embedding.weight'
            
            # Pre/post layer norms
            new_key = new_key.replace('vision_model.ln_pre.', 'vision_model.pre_layrnorm.')
            new_key = new_key.replace('vision_model.ln_post.', 'vision_model.post_layernorm.')
            
            # Transformer blocks
            new_key = new_key.replace('vision_model.transformer.resblocks.', 'vision_model.encoder.layers.')
            
            # Attention layers - split QKV
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
            
            # MLP layers
            new_key = new_key.replace('.mlp.c_fc.', '.mlp.fc1.')
            new_key = new_key.replace('.mlp.c_proj.', '.mlp.fc2.')
            
            # Layer norms
            new_key = new_key.replace('.ln_1.', '.layer_norm1.')
            new_key = new_key.replace('.ln_2.', '.layer_norm2.')
            
            # Vision projection
            if new_key == 'vision_model.proj':
                new_key = 'visual_projection.weight'
                value = value.T
        
        # Handle text model keys
        elif key.startswith('transformer.') or key in ['token_embedding.weight', 'positional_embedding', 
                                                         'ln_final.weight', 'ln_final.bias', 'text_projection']:
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


@dataclass
class EmbeddingStats:
    """Statistics for embedding comparison"""
    dataset_name: str
    num_samples: int
    
    # SAE embedding stats
    sae_cosine_mean: float
    sae_cosine_std: float
    sae_cosine_min: float
    sae_cosine_max: float
    sae_l2_mean: float
    sae_l2_std: float
    sae_sparsity_base_mean: float
    sae_sparsity_lora_mean: float
    
    # CLIP embedding stats
    clip_cosine_mean: float
    clip_cosine_std: float
    clip_cosine_min: float
    clip_cosine_max: float
    clip_l2_mean: float
    clip_l2_std: float


def get_sae_encoding(sae, x):
    """
    Get SAE encoding using the correct method.
    Handles different SAE implementations.
    """
    # Try forward pass - SAE typically returns (reconstruction, latent) or just latent
    with torch.no_grad():
        out = sae(x)
        if isinstance(out, tuple):
            # Common pattern: (reconstruction, latent, ...) or (latent, ...)
            # Check which one looks like the sparse encoding (higher dimension usually)
            if len(out) >= 2:
                # Usually the second element is the latent/encoded representation
                return out[1]
            return out[0]
        return out


def load_ucf101_frames(data_root: str, max_samples: int = 100):
    """
    Load UCF101 dataset frames.
    UCF101 is a video dataset, so we extract frames from videos.
    """
    from pathlib import Path
    import random
    
    data_path = Path(data_root)
    
    # Common UCF101 locations to try
    possible_paths = [
        data_path / "UCF101" / "UCF-101",
        data_path / "ucf101" / "UCF-101", 
        data_path / "UCF-101",
        data_path / "ucf101",
        Path("/home/sunayana/Documents/Concept_LoRA/data/ucf101"),
        Path("/home/sunayana/data/ucf101"),
    ]
    
    ucf_path = None
    for p in possible_paths:
        if p.exists():
            ucf_path = p
            print(f"   Found UCF101 at: {ucf_path}")
            break
    
    if ucf_path is None:
        print("   UCF101 not found in standard locations.")
        print("   Trying to load from HuggingFace datasets...")
        return load_ucf101_from_huggingface(max_samples)
    
    # Load frames from video folders
    frames = []
    classes = sorted([d for d in ucf_path.iterdir() if d.is_dir()])
    
    for class_dir in classes[:20]:  # Limit classes for speed
        video_files = list(class_dir.glob("*.avi")) + list(class_dir.glob("*.mp4"))
        for video_file in video_files[:5]:  # Limit videos per class
            # Extract a frame from the video
            try:
                import cv2
                cap = cv2.VideoCapture(str(video_file))
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame))
                cap.release()
            except Exception as e:
                continue
            
            if len(frames) >= max_samples:
                break
        if len(frames) >= max_samples:
            break
    
    return frames


def load_ucf101_from_huggingface(max_samples: int = 100):
    """Load UCF101 from HuggingFace datasets"""
    try:
        from datasets import load_dataset
        print("   Loading UCF101 from HuggingFace...")
        dataset = load_dataset("sayakpaul/ucf101-subset", split="train")
        
        frames = []
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break
            if 'video' in sample:
                # Get first frame from video
                video = sample['video']
                if hasattr(video, '__getitem__'):
                    frames.append(video[0])
                else:
                    frames.append(video)
            elif 'image' in sample:
                frames.append(sample['image'])
        
        return frames
    except Exception as e:
        print(f"   Failed to load from HuggingFace: {e}")
        return []


def load_sample_images(image_dir: str, max_samples: int = 100):
    """Load sample images from a directory"""
    from pathlib import Path
    
    image_dir = Path(image_dir)
    if not image_dir.exists():
        return []
    
    images = []
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    
    for ext in extensions:
        for img_path in image_dir.rglob(ext):
            try:
                img = Image.open(img_path).convert('RGB')
                images.append(img)
                if len(images) >= max_samples:
                    break
            except Exception:
                continue
        if len(images) >= max_samples:
            break
    
    return images


class EmbeddingComparator:
    """Compare embeddings between base and LoRA models"""
    
    def __init__(
        self,
        sae_path: str,
        lora_weights_path: str,
        device: str = "cpu",
        backbone: str = "openai/clip-vit-base-patch16",
    ):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(backbone)
        
        print("=" * 80)
        print("Initializing Embedding Comparator")
        print("=" * 80)
        
        # Load base model with SAE
        print("\n1. Loading BASE model...")
        self.sae, self.vit_base, self.cfg = get_sae_and_vit(
            sae_path=sae_path,
            vit_type="base",
            device=device,
            backbone=backbone,
            model_path=None,
            classnames=None,
        )
        print("   ✓ Base model loaded")
        
        # Debug: Print SAE info
        print(f"   SAE type: {type(self.sae).__name__}")
        
        # Create LoRA model (separate instance, then load weights)
        print("\n2. Loading LoRA model...")
        self.sae_lora, self.vit_lora, _ = get_sae_and_vit(
            sae_path=sae_path,
            vit_type="base",
            device=device,
            backbone=backbone,
            model_path=None,
            classnames=None,
        )
        
        # Load and convert LoRA weights
        state_dict = torch.load(lora_weights_path, map_location=device)
        print(f"   Loaded checkpoint with {len(state_dict)} keys")
        
        hf_state_dict = convert_openai_to_hf_clip(state_dict)
        print(f"   Converted to {len(hf_state_dict)} HuggingFace keys")
        
        incomp = self.vit_lora.model.load_state_dict(hf_state_dict, strict=False)
        print(f"   Missing keys: {len(incomp.missing_keys)}, Unexpected: {len(incomp.unexpected_keys)}")
        
        # Verify weights differ
        self._verify_weights_differ()
        
        # Try to load available datasets
        print("\n3. Loading datasets...")
        try:
            self.datasets = load_datasets()
            print(f"   ✓ Loaded {len(self.datasets)} datasets")
            print(f"   Available: {list(self.datasets.keys())}")
        except Exception as e:
            print(f"   Could not load datasets: {e}")
            self.datasets = {}
        
        print("\n" + "=" * 80)
        print("Initialization Complete")
        print("=" * 80 + "\n")
    
    def _verify_weights_differ(self):
        """Verify LoRA weights actually differ from base"""
        with torch.no_grad():
            base_w = self.vit_base.model.vision_model.encoder.layers[0].self_attn.q_proj.weight
            lora_w = self.vit_lora.model.vision_model.encoder.layers[0].self_attn.q_proj.weight
            max_diff = (base_w - lora_w).abs().max().item()
            
            if max_diff > 1e-6:
                print(f"   ✓ LoRA weights differ from base (max diff: {max_diff:.6f})")
            else:
                print(f"   ⚠️ WARNING: LoRA weights appear identical to base!")
    
    @torch.no_grad()
    def extract_embeddings(self, vit, sae, image_tensor):
        """Extract CLIP and SAE embeddings for an image"""
        # Get CLIP embeddings (patch tokens from last layer)
        outputs = vit.model.vision_model(
            pixel_values=image_tensor,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Get patch embeddings (excluding CLS token)
        last_hidden = outputs.last_hidden_state  # [B, 197, 768]
        patch_embeddings = last_hidden[:, 1:, :]  # [B, 196, 768] - exclude CLS
        
        # Get SAE embeddings
        patch_flat = patch_embeddings.reshape(-1, patch_embeddings.shape[-1])  # [B*196, 768]
        sae_embeddings = get_sae_encoding(sae, patch_flat)  # [B*196, sae_dim]
        
        return patch_embeddings, sae_embeddings
    
    def compare_single_image(self, image_tensor):
        """Compare embeddings for a single image"""
        # Extract from base model
        base_clip, base_sae = self.extract_embeddings(self.vit_base, self.sae, image_tensor)
        
        # Extract from LoRA model  
        lora_clip, lora_sae = self.extract_embeddings(self.vit_lora, self.sae_lora, image_tensor)
        
        # Flatten for comparison
        base_clip_flat = base_clip.flatten()
        lora_clip_flat = lora_clip.flatten()
        base_sae_flat = base_sae.flatten()
        lora_sae_flat = lora_sae.flatten()
        
        # Compute metrics
        sae_cosine = F.cosine_similarity(
            base_sae_flat.unsqueeze(0), 
            lora_sae_flat.unsqueeze(0)
        ).item()
        sae_l2 = torch.norm(base_sae_flat - lora_sae_flat, p=2).item()
        
        clip_cosine = F.cosine_similarity(
            base_clip_flat.unsqueeze(0), 
            lora_clip_flat.unsqueeze(0)
        ).item()
        clip_l2 = torch.norm(base_clip_flat - lora_clip_flat, p=2).item()
        
        # Sparsity (fraction of near-zero activations)
        sae_sparsity_base = (base_sae_flat.abs() < 1e-5).float().mean().item()
        sae_sparsity_lora = (lora_sae_flat.abs() < 1e-5).float().mean().item()
        
        return {
            'sae_cosine': sae_cosine,
            'sae_l2': sae_l2,
            'clip_cosine': clip_cosine,
            'clip_l2': clip_l2,
            'sae_sparsity_base': sae_sparsity_base,
            'sae_sparsity_lora': sae_sparsity_lora,
        }
    
    def compare_images(
        self, 
        images: List,
        dataset_name: str = "custom",
    ) -> EmbeddingStats:
        """Compare embeddings for a list of images"""
        
        # Collect per-sample metrics
        sae_cosines = []
        sae_l2s = []
        clip_cosines = []
        clip_l2s = []
        sae_sparsity_bases = []
        sae_sparsity_loras = []
        
        for i, image in enumerate(tqdm(images, desc=f"Processing {dataset_name}")):
            try:
                # Preprocess image
                inputs = self.processor(images=image, return_tensors="pt")
                image_tensor = inputs['pixel_values'].to(self.device)
                
                metrics = self.compare_single_image(image_tensor)
                sae_cosines.append(metrics['sae_cosine'])
                sae_l2s.append(metrics['sae_l2'])
                clip_cosines.append(metrics['clip_cosine'])
                clip_l2s.append(metrics['clip_l2'])
                sae_sparsity_bases.append(metrics['sae_sparsity_base'])
                sae_sparsity_loras.append(metrics['sae_sparsity_lora'])
            except Exception as e:
                print(f"   Warning: Failed on sample {i}: {e}")
                continue
        
        # Handle empty results
        if len(sae_cosines) == 0:
            print(f"   ⚠️ No valid samples processed for {dataset_name}")
            return EmbeddingStats(
                dataset_name=dataset_name,
                num_samples=0,
                sae_cosine_mean=0.0, sae_cosine_std=0.0, sae_cosine_min=0.0, sae_cosine_max=0.0,
                sae_l2_mean=0.0, sae_l2_std=0.0,
                sae_sparsity_base_mean=0.0, sae_sparsity_lora_mean=0.0,
                clip_cosine_mean=0.0, clip_cosine_std=0.0, clip_cosine_min=0.0, clip_cosine_max=0.0,
                clip_l2_mean=0.0, clip_l2_std=0.0,
            )
        
        # Compute statistics
        return EmbeddingStats(
            dataset_name=dataset_name,
            num_samples=len(sae_cosines),
            sae_cosine_mean=float(np.mean(sae_cosines)),
            sae_cosine_std=float(np.std(sae_cosines)),
            sae_cosine_min=float(np.min(sae_cosines)),
            sae_cosine_max=float(np.max(sae_cosines)),
            sae_l2_mean=float(np.mean(sae_l2s)),
            sae_l2_std=float(np.std(sae_l2s)),
            sae_sparsity_base_mean=float(np.mean(sae_sparsity_bases)),
            sae_sparsity_lora_mean=float(np.mean(sae_sparsity_loras)),
            clip_cosine_mean=float(np.mean(clip_cosines)),
            clip_cosine_std=float(np.std(clip_cosines)),
            clip_cosine_min=float(np.min(clip_cosines)),
            clip_cosine_max=float(np.max(clip_cosines)),
            clip_l2_mean=float(np.mean(clip_l2s)),
            clip_l2_std=float(np.std(clip_l2s)),
        )


def save_results(results: List[EmbeddingStats], output_dir: str):
    """Save comparison results"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not results:
        print("No results to save!")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame([asdict(r) for r in results])
    
    # Save CSV
    csv_path = output_path / "embedding_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")
    
    # Save JSON summary
    summary = {
        'total_datasets': len(results),
        'total_samples': int(df['num_samples'].sum()),
        'overall_stats': {
            'sae_cosine_mean': float(df['sae_cosine_mean'].mean()),
            'sae_cosine_std': float(df['sae_cosine_mean'].std()) if len(df) > 1 else 0.0,
            'clip_cosine_mean': float(df['clip_cosine_mean'].mean()),
            'clip_cosine_std': float(df['clip_cosine_mean'].std()) if len(df) > 1 else 0.0,
            'sae_l2_mean': float(df['sae_l2_mean'].mean()),
            'clip_l2_mean': float(df['clip_l2_mean'].mean()),
        },
        'per_dataset': df.to_dict(orient='records'),
    }
    
    json_path = output_path / "summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved JSON to {json_path}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare SAE/CLIP embeddings between base and LoRA models')
    parser.add_argument('--sae-path', type=str, required=True, help='Path to SAE weights')
    parser.add_argument('--lora-path', type=str, required=True, help='Path to LoRA merged weights')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--max-samples', type=int, default=100, help='Max samples per dataset')
    parser.add_argument('--image-dir', type=str, default=None, help='Directory with images to process')
    parser.add_argument('--output-dir', type=str, default='results/embedding_comparison', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = EmbeddingComparator(
        sae_path=args.sae_path,
        lora_weights_path=args.lora_path,
        device=args.device,
    )
    
    results = []
    
    # Try to load images
    if args.image_dir:
        print(f"\nLoading images from: {args.image_dir}")
        images = load_sample_images(args.image_dir, args.max_samples)
        if images:
            print(f"   ✓ Loaded {len(images)} images")
            stats = comparator.compare_images(images, dataset_name="ucf101")
            results.append(stats)
        else:
            print(f"   ⚠️ No images found in {args.image_dir}")
    else:
        # Try UCF101
        print("\nTrying to load UCF101 frames...")
        images = load_ucf101_frames("data", args.max_samples)
        
        if not images:
            # Fallback: use sample images from patchsae data folder
            print("\nFalling back to sample images...")
            sample_dirs = [
                "data/image",
                "/home/sunayana/Documents/Concept_LoRA/patchsae/data/image",
            ]
            for sample_dir in sample_dirs:
                images = load_sample_images(sample_dir, args.max_samples)
                if images:
                    print(f"   ✓ Loaded {len(images)} images from {sample_dir}")
                    break
        
        if images:
            print(f"   ✓ Total images to process: {len(images)}")
            stats = comparator.compare_images(images, dataset_name="ucf101")
            results.append(stats)
        else:
            print("\n⚠️ No images found! Please specify --image-dir")
            print("Example: python compare_embeddings.py --sae-path ... --lora-path ... --image-dir /path/to/images")
            return
    
    # Save results
    if results:
        save_results(results, args.output_dir)
        
        print("\n" + "=" * 80)
        print("COMPARISON COMPLETE")
        print("=" * 80)
        for r in results:
            print(f"\nDataset: {r.dataset_name}")
            print(f"  Samples: {r.num_samples}")
            print(f"  SAE Cosine Similarity: {r.sae_cosine_mean:.4f} ± {r.sae_cosine_std:.4f}")
            print(f"  CLIP Cosine Similarity: {r.clip_cosine_mean:.4f} ± {r.clip_cosine_std:.4f}")
            print(f"  SAE L2 Distance: {r.sae_l2_mean:.4f} ± {r.sae_l2_std:.4f}")
            print(f"  CLIP L2 Distance: {r.clip_l2_mean:.4f} ± {r.clip_l2_std:.4f}")
        print("=" * 80)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()