"""
Comprehensive comparison script for SAE embeddings and CLIP activations
between base CLIP model and LoRA fine-tuned model.

This script:
1. Loads both base and LoRA models
2. Processes entire specified dataset
3. Compares SAE embeddings and CLIP activations
4. Generates detailed comparison statistics and visualizations
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import OrderedDict, defaultdict
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import pandas as pd

from src.demo.core import SAETester
from tasks.utils import (
    get_all_classnames,
    get_max_acts_and_images,
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
        
        # Logit scale
        elif key == 'logit_scale':
            new_key = 'logit_scale'
        
        if new_key:
            hf_state_dict[new_key] = value
        else:
            print(f"Warning: Unhandled key: {key}")
    
    return hf_state_dict


@dataclass
class ComparisonMetrics:
    """Metrics for comparing model activations"""
    # SAE metrics
    sae_cosine_similarity: float
    sae_l2_distance: float
    sae_l1_distance: float
    sae_max_difference: float
    sae_mean_difference: float
    sae_sparsity_base: float
    sae_sparsity_lora: float
    
    # CLIP activation metrics
    clip_cosine_similarity: float
    clip_l2_distance: float
    clip_l1_distance: float
    clip_max_difference: float
    clip_mean_difference: float
    
    # Per-layer metrics
    layer_wise_similarities: Dict[int, float]
    layer_wise_l2_distances: Dict[int, float]
    
    # Image metadata
    image_id: str
    dataset_name: str


class ModelComparator:
    """Compares SAE embeddings and CLIP activations between base and LoRA models"""
    
    def __init__(
        self,
        sae_path: str,
        base_backbone: str = "openai/clip-vit-base-patch16",
        lora_weights_path: Optional[str] = None,
        device: str = "cpu",
        feature_data_root: str = "out/feature_data",
        sae_runname: str = "sae_base",
        vit_name: str = "base",
    ):
        """
        Initialize the comparator with both base and LoRA models.
        
        Args:
            sae_path: Path to SAE weights
            base_backbone: HuggingFace model identifier for base CLIP
            lora_weights_path: Path to merged LoRA weights (OpenAI format)
            device: Device to run on
            feature_data_root: Root directory for feature data
            sae_runname: SAE run name
            vit_name: ViT variant name
        """
        self.device = device
        self.feature_data_root = feature_data_root
        self.sae_runname = sae_runname
        self.vit_name = vit_name
        
        print("=" * 80)
        print("Initializing Model Comparator")
        print("=" * 80)
        
        # Load base model with SAE
        print("\n1. Loading BASE model...")
        self.sae_base, self.vit_base, self.cfg = get_sae_and_vit(
            sae_path=sae_path,
            vit_type="base",
            device=device,
            backbone=base_backbone,
            model_path=None,
            classnames=None,
        )
        print("✓ Base model loaded")
        
        # Load datasets and precomputed activations
        print("\n2. Loading datasets...")
        self.datasets = load_datasets()
        self.max_act_imgs, self.mean_acts = get_max_acts_and_images(
            self.datasets, feature_data_root, sae_runname, vit_name
        )
        print(f"✓ Loaded {len(self.datasets)} datasets")
        
        # Load LoRA model if provided
        if lora_weights_path:
            print(f"\n3. Loading LoRA model from {lora_weights_path}...")
            
            # Create a copy of base model for LoRA
            self.sae_lora, self.vit_lora, _ = get_sae_and_vit(
                sae_path=sae_path,
                vit_type="base",
                device=device,
                backbone=base_backbone,
                model_path=None,
                classnames=None,
            )
            
            # Load and convert LoRA weights
            state_dict = torch.load(lora_weights_path, map_location=device)
            print(f"   Loaded {len(state_dict)} keys from checkpoint")
            
            hf_state_dict = convert_openai_to_hf_clip(state_dict)
            print(f"   Converted to {len(hf_state_dict)} HuggingFace keys")
            
            # Load weights into LoRA model
            incomp = self.vit_lora.model.load_state_dict(hf_state_dict, strict=False)
            
            if len(incomp.missing_keys) > 0 or len(incomp.unexpected_keys) > 0:
                print(f"   ⚠️  Missing keys: {len(incomp.missing_keys)}")
                print(f"   ⚠️  Unexpected keys: {len(incomp.unexpected_keys)}")
            else:
                print("   ✓ All weights loaded successfully")
            
            # Verify that LoRA weights actually differ from base
            self._verify_lora_weights()
            
            self.has_lora = True
            print("✓ LoRA model loaded and verified")
        else:
            self.sae_lora = None
            self.vit_lora = None
            self.has_lora = False
            print("\n3. No LoRA model provided - will only analyze base model")
        
        # Load class names
        self.classnames = get_all_classnames(self.datasets, data_root="configs/classnames")
        
        print("\n" + "=" * 80)
        print("Initialization Complete")
        print("=" * 80 + "\n")
    
    def _verify_lora_weights(self):
        """Verify that LoRA weights actually differ from base weights"""
        with torch.no_grad():
            w_base = self.vit_base.model.vision_model.encoder.layers[0].self_attn.q_proj.weight
            w_lora = self.vit_lora.model.vision_model.encoder.layers[0].self_attn.q_proj.weight
            max_diff = (w_base - w_lora).abs().max().item()
            
            print(f"   Weight difference check (layer 0, q_proj):")
            print(f"   Max |diff|: {max_diff:.8f}")
            
            if max_diff < 1e-6:
                print("   ⚠️  WARNING: LoRA weights appear identical to base!")
                print("   This may indicate the weights weren't properly merged.")
            else:
                print("   ✓ LoRA weights differ from base")
    
    @torch.no_grad()
    def extract_activations(self, model, sae, image_tensor):
        """
        Extract both CLIP activations and SAE embeddings for an image.
        
        Returns:
            tuple: (clip_activations, sae_embeddings, layer_activations)
        """
        model.model.eval()
        sae.eval()
        
        # Get CLIP vision features
        outputs = model.model.vision_model(pixel_values=image_tensor)
        
        # Get patch features (excluding CLS token)
        patch_features = outputs.last_hidden_state[:, 1:, :]  # [B, num_patches, hidden_dim]
        
        # Get CLS token
        cls_token = outputs.last_hidden_state[:, 0, :]  # [B, hidden_dim]
        
        # Get SAE embeddings
        batch_size, num_patches, hidden_dim = patch_features.shape
        flat_features = patch_features.reshape(-1, hidden_dim)  # [B*num_patches, hidden_dim]
        
        # Pass through SAE
        sae_output = sae(flat_features)
        sae_embeddings = sae_output.reshape(batch_size, num_patches, -1)  # [B, num_patches, sae_dim]
        
        # Get layer-wise activations for more detailed comparison
        layer_activations = {}
        for i, layer in enumerate(model.model.vision_model.encoder.layers):
            if i == 0:
                layer_input = model.model.vision_model.embeddings(image_tensor)
            else:
                # Use outputs from previous layer
                layer_input = layer_activations[i-1]
            
            layer_output = layer(layer_input)[0]
            layer_activations[i] = layer_output
        
        return cls_token, patch_features, sae_embeddings, layer_activations
    
    def compute_metrics(
        self,
        base_clip: torch.Tensor,
        lora_clip: torch.Tensor,
        base_sae: torch.Tensor,
        lora_sae: torch.Tensor,
        base_layers: Dict[int, torch.Tensor],
        lora_layers: Dict[int, torch.Tensor],
        image_id: str,
        dataset_name: str,
    ) -> ComparisonMetrics:
        """Compute comprehensive comparison metrics"""
        
        # Flatten tensors for comparison
        base_clip_flat = base_clip.flatten()
        lora_clip_flat = lora_clip.flatten()
        base_sae_flat = base_sae.flatten()
        lora_sae_flat = lora_sae.flatten()
        
        # SAE metrics
        sae_cosine = F.cosine_similarity(base_sae_flat.unsqueeze(0), lora_sae_flat.unsqueeze(0)).item()
        sae_l2 = torch.norm(base_sae_flat - lora_sae_flat, p=2).item()
        sae_l1 = torch.norm(base_sae_flat - lora_sae_flat, p=1).item()
        sae_max_diff = (base_sae_flat - lora_sae_flat).abs().max().item()
        sae_mean_diff = (base_sae_flat - lora_sae_flat).abs().mean().item()
        
        # SAE sparsity (percentage of near-zero activations)
        sae_sparsity_base = (base_sae_flat.abs() < 1e-5).float().mean().item()
        sae_sparsity_lora = (lora_sae_flat.abs() < 1e-5).float().mean().item()
        
        # CLIP metrics
        clip_cosine = F.cosine_similarity(base_clip_flat.unsqueeze(0), lora_clip_flat.unsqueeze(0)).item()
        clip_l2 = torch.norm(base_clip_flat - lora_clip_flat, p=2).item()
        clip_l1 = torch.norm(base_clip_flat - lora_clip_flat, p=1).item()
        clip_max_diff = (base_clip_flat - lora_clip_flat).abs().max().item()
        clip_mean_diff = (base_clip_flat - lora_clip_flat).abs().mean().item()
        
        # Layer-wise metrics
        layer_sims = {}
        layer_l2s = {}
        for layer_idx in base_layers.keys():
            base_layer = base_layers[layer_idx].flatten()
            lora_layer = lora_layers[layer_idx].flatten()
            
            layer_sims[layer_idx] = F.cosine_similarity(
                base_layer.unsqueeze(0), lora_layer.unsqueeze(0)
            ).item()
            layer_l2s[layer_idx] = torch.norm(base_layer - lora_layer, p=2).item()
        
        return ComparisonMetrics(
            sae_cosine_similarity=sae_cosine,
            sae_l2_distance=sae_l2,
            sae_l1_distance=sae_l1,
            sae_max_difference=sae_max_diff,
            sae_mean_difference=sae_mean_diff,
            sae_sparsity_base=sae_sparsity_base,
            sae_sparsity_lora=sae_sparsity_lora,
            clip_cosine_similarity=clip_cosine,
            clip_l2_distance=clip_l2,
            clip_l1_distance=clip_l1,
            clip_max_difference=clip_max_diff,
            clip_mean_difference=clip_mean_diff,
            layer_wise_similarities=layer_sims,
            layer_wise_l2_distances=layer_l2s,
            image_id=image_id,
            dataset_name=dataset_name,
        )
    
    def compare_on_dataset(
        self,
        dataset_name: str,
        max_samples: Optional[int] = None,
        batch_size: int = 1,
    ) -> List[ComparisonMetrics]:
        """
        Compare models on entire dataset.
        
        Args:
            dataset_name: Name of dataset to process
            max_samples: Maximum number of samples to process (None for all)
            batch_size: Batch size for processing
            
        Returns:
            List of ComparisonMetrics for each sample
        """
        if not self.has_lora:
            raise ValueError("LoRA model not loaded - cannot perform comparison")
        
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found. Available: {list(self.datasets.keys())}")
        
        dataset = self.datasets[dataset_name]
        num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        
        print(f"\nProcessing {num_samples} samples from {dataset_name}...")
        
        all_metrics = []
        
        for idx in tqdm(range(num_samples), desc=f"Comparing on {dataset_name}"):
            # Get image
            sample = dataset[idx]
            if isinstance(sample, (tuple, list)):
                image = sample[0]
            else:
                image = sample
            
            # Prepare image tensor
            if hasattr(self.vit_base, 'preprocess'):
                image_tensor = self.vit_base.preprocess(image).unsqueeze(0).to(self.device)
            else:
                # Assume image is already preprocessed
                image_tensor = image.unsqueeze(0).to(self.device) if len(image.shape) == 3 else image.to(self.device)
            
            # Extract activations from both models
            base_cls, base_patches, base_sae, base_layers = self.extract_activations(
                self.vit_base, self.sae_base, image_tensor
            )
            lora_cls, lora_patches, lora_sae, lora_layers = self.extract_activations(
                self.vit_lora, self.sae_lora, image_tensor
            )
            
            # Compute metrics
            metrics = self.compute_metrics(
                base_patches, lora_patches,
                base_sae, lora_sae,
                base_layers, lora_layers,
                image_id=f"{dataset_name}_{idx}",
                dataset_name=dataset_name,
            )
            
            all_metrics.append(metrics)
        
        return all_metrics
    
    def compare_on_all_datasets(
        self,
        max_samples_per_dataset: Optional[int] = None,
        dataset_filter: Optional[List[str]] = None,
    ) -> Dict[str, List[ComparisonMetrics]]:
        """
        Compare models across all datasets.
        
        Args:
            max_samples_per_dataset: Max samples per dataset
            dataset_filter: List of dataset names to process (None for all)
            
        Returns:
            Dictionary mapping dataset names to lists of metrics
        """
        if not self.has_lora:
            raise ValueError("LoRA model not loaded - cannot perform comparison")
        
        datasets_to_process = dataset_filter if dataset_filter else list(self.datasets.keys())
        
        all_results = {}
        
        print("\n" + "=" * 80)
        print("Starting Multi-Dataset Comparison")
        print("=" * 80)
        
        for dataset_name in datasets_to_process:
            if dataset_name in self.datasets:
                try:
                    metrics = self.compare_on_dataset(
                        dataset_name,
                        max_samples=max_samples_per_dataset
                    )
                    all_results[dataset_name] = metrics
                    print(f"✓ Completed {dataset_name}: {len(metrics)} samples")
                except Exception as e:
                    print(f"✗ Error processing {dataset_name}: {e}")
        
        print("\n" + "=" * 80)
        print("Comparison Complete")
        print("=" * 80 + "\n")
        
        return all_results


class ComparisonAnalyzer:
    """Analyze and visualize comparison results"""
    
    def __init__(self, results: Dict[str, List[ComparisonMetrics]]):
        self.results = results
        self.df = self._results_to_dataframe()
    
    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        records = []
        for dataset_name, metrics_list in self.results.items():
            for metrics in metrics_list:
                record = asdict(metrics)
                # Flatten layer-wise metrics
                for layer_idx, sim in record['layer_wise_similarities'].items():
                    record[f'layer_{layer_idx}_similarity'] = sim
                for layer_idx, l2 in record['layer_wise_l2_distances'].items():
                    record[f'layer_{layer_idx}_l2'] = l2
                
                del record['layer_wise_similarities']
                del record['layer_wise_l2_distances']
                
                records.append(record)
        
        return pd.DataFrame(records)
    
    def generate_summary_statistics(self) -> pd.DataFrame:
        """Generate summary statistics across all datasets"""
        summary_cols = [
            'sae_cosine_similarity', 'sae_l2_distance', 'sae_mean_difference',
            'sae_sparsity_base', 'sae_sparsity_lora',
            'clip_cosine_similarity', 'clip_l2_distance', 'clip_mean_difference'
        ]
        
        summary = self.df.groupby('dataset_name')[summary_cols].agg(['mean', 'std', 'min', 'max'])
        return summary
    
    def plot_comparison_distributions(self, output_path: str = "comparison_distributions.png"):
        """Plot distributions of key metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Distribution of Comparison Metrics', fontsize=16)
        
        metrics_to_plot = [
            ('sae_cosine_similarity', 'SAE Cosine Similarity'),
            ('sae_l2_distance', 'SAE L2 Distance'),
            ('sae_mean_difference', 'SAE Mean Difference'),
            ('clip_cosine_similarity', 'CLIP Cosine Similarity'),
            ('clip_l2_distance', 'CLIP L2 Distance'),
            ('clip_mean_difference', 'CLIP Mean Difference'),
        ]
        
        for idx, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            
            for dataset_name in self.df['dataset_name'].unique():
                dataset_df = self.df[self.df['dataset_name'] == dataset_name]
                ax.hist(dataset_df[metric], alpha=0.5, label=dataset_name, bins=30)
            
            ax.set_xlabel(title)
            ax.set_ylabel('Frequency')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved distribution plot to {output_path}")
        plt.close()
    
    def plot_layer_wise_comparison(self, output_path: str = "layer_wise_comparison.png"):
        """Plot layer-wise similarity and distance trends"""
        # Extract layer-wise columns
        layer_sim_cols = [col for col in self.df.columns if col.startswith('layer_') and col.endswith('_similarity')]
        layer_l2_cols = [col for col in self.df.columns if col.startswith('layer_') and col.endswith('_l2')]
        
        if not layer_sim_cols:
            print("No layer-wise data found")
            return
        
        # Extract layer numbers
        layer_numbers = sorted([int(col.split('_')[1]) for col in layer_sim_cols])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Layer-wise Comparison Metrics', fontsize=16)
        
        # Plot similarities
        for dataset_name in self.df['dataset_name'].unique():
            dataset_df = self.df[self.df['dataset_name'] == dataset_name]
            mean_sims = [dataset_df[f'layer_{i}_similarity'].mean() for i in layer_numbers]
            std_sims = [dataset_df[f'layer_{i}_similarity'].std() for i in layer_numbers]
            
            ax1.plot(layer_numbers, mean_sims, marker='o', label=dataset_name, linewidth=2)
            ax1.fill_between(
                layer_numbers,
                [m - s for m, s in zip(mean_sims, std_sims)],
                [m + s for m, s in zip(mean_sims, std_sims)],
                alpha=0.2
            )
        
        ax1.set_xlabel('Layer Number')
        ax1.set_ylabel('Cosine Similarity')
        ax1.set_title('Layer-wise Cosine Similarity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot L2 distances
        for dataset_name in self.df['dataset_name'].unique():
            dataset_df = self.df[self.df['dataset_name'] == dataset_name]
            mean_l2s = [dataset_df[f'layer_{i}_l2'].mean() for i in layer_numbers]
            std_l2s = [dataset_df[f'layer_{i}_l2'].std() for i in layer_numbers]
            
            ax2.plot(layer_numbers, mean_l2s, marker='s', label=dataset_name, linewidth=2)
            ax2.fill_between(
                layer_numbers,
                [m - s for m, s in zip(mean_l2s, std_l2s)],
                [m + s for m, s in zip(mean_l2s, std_l2s)],
                alpha=0.2
            )
        
        ax2.set_xlabel('Layer Number')
        ax2.set_ylabel('L2 Distance')
        ax2.set_title('Layer-wise L2 Distance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved layer-wise comparison to {output_path}")
        plt.close()
    
    def plot_sae_vs_clip_correlation(self, output_path: str = "sae_clip_correlation.png"):
        """Plot correlation between SAE and CLIP metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('SAE vs CLIP Metrics Correlation', fontsize=16)
        
        # Cosine similarity correlation
        ax1 = axes[0]
        for dataset_name in self.df['dataset_name'].unique():
            dataset_df = self.df[self.df['dataset_name'] == dataset_name]
            ax1.scatter(
                dataset_df['sae_cosine_similarity'],
                dataset_df['clip_cosine_similarity'],
                alpha=0.6,
                label=dataset_name,
                s=50
            )
        
        ax1.set_xlabel('SAE Cosine Similarity')
        ax1.set_ylabel('CLIP Cosine Similarity')
        ax1.set_title('Cosine Similarity Correlation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add diagonal line
        lims = [
            np.min([ax1.get_xlim(), ax1.get_ylim()]),
            np.max([ax1.get_xlim(), ax1.get_ylim()]),
        ]
        ax1.plot(lims, lims, 'k--', alpha=0.3, zorder=0)
        
        # L2 distance correlation
        ax2 = axes[1]
        for dataset_name in self.df['dataset_name'].unique():
            dataset_df = self.df[self.df['dataset_name'] == dataset_name]
            ax2.scatter(
                dataset_df['sae_l2_distance'],
                dataset_df['clip_l2_distance'],
                alpha=0.6,
                label=dataset_name,
                s=50
            )
        
        ax2.set_xlabel('SAE L2 Distance')
        ax2.set_ylabel('CLIP L2 Distance')
        ax2.set_title('L2 Distance Correlation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved SAE-CLIP correlation to {output_path}")
        plt.close()
    
    def save_results(self, output_dir: str):
        """Save all results and visualizations"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving results to {output_dir}...")
        
        # Save summary statistics
        summary = self.generate_summary_statistics()
        summary.to_csv(output_dir / "summary_statistics.csv")
        print(f"✓ Saved summary statistics")
        
        # Save full dataframe
        self.df.to_csv(output_dir / "full_results.csv", index=False)
        print(f"✓ Saved full results CSV")
        
        # Save plots
        self.plot_comparison_distributions(str(output_dir / "comparison_distributions.png"))
        self.plot_layer_wise_comparison(str(output_dir / "layer_wise_comparison.png"))
        self.plot_sae_vs_clip_correlation(str(output_dir / "sae_clip_correlation.png"))
        
        # Save summary JSON
        summary_dict = {
            'num_datasets': len(self.results),
            'total_samples': len(self.df),
            'datasets': {
                dataset_name: len(metrics)
                for dataset_name, metrics in self.results.items()
            },
            'global_means': {
                'sae_cosine_similarity': float(self.df['sae_cosine_similarity'].mean()),
                'clip_cosine_similarity': float(self.df['clip_cosine_similarity'].mean()),
                'sae_l2_distance': float(self.df['sae_l2_distance'].mean()),
                'clip_l2_distance': float(self.df['clip_l2_distance'].mean()),
            }
        }
        
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary_dict, f, indent=2)
        print(f"✓ Saved summary JSON")
        
        print(f"\n✓ All results saved to {output_dir}")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare SAE embeddings and CLIP activations')
    parser.add_argument('--sae-path', type=str, required=True, help='Path to SAE weights')
    parser.add_argument('--lora-path', type=str, required=True, help='Path to LoRA merged weights')
    parser.add_argument('--backbone', type=str, default='openai/clip-vit-base-patch16', 
                       help='HuggingFace model identifier')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--max-samples', type=int, default=None, 
                       help='Max samples per dataset')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help='Specific datasets to process')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = ModelComparator(
        sae_path=args.sae_path,
        base_backbone=args.backbone,
        lora_weights_path=args.lora_path,
        device=args.device,
    )
    
    # Run comparison
    results = comparator.compare_on_all_datasets(
        max_samples_per_dataset=args.max_samples,
        dataset_filter=args.datasets,
    )
    
    # Analyze and save results
    analyzer = ComparisonAnalyzer(results)
    analyzer.save_results(args.output_dir)
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Total samples processed: {len(analyzer.df)}")
    print(f"Datasets processed: {len(results)}")
    
    # Print quick summary
    print("\n" + "-" * 80)
    print("QUICK SUMMARY")
    print("-" * 80)
    print(f"Mean SAE Cosine Similarity: {analyzer.df['sae_cosine_similarity'].mean():.4f}")
    print(f"Mean CLIP Cosine Similarity: {analyzer.df['clip_cosine_similarity'].mean():.4f}")
    print(f"Mean SAE L2 Distance: {analyzer.df['sae_l2_distance'].mean():.4f}")
    print(f"Mean CLIP L2 Distance: {analyzer.df['clip_l2_distance'].mean():.4f}")
    print("-" * 80)


if __name__ == "__main__":
    main()