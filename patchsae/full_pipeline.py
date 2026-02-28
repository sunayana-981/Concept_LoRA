import os
import sys
patchsae_dir = os.path.join(os.path.dirname(__file__), 'patchsae')
sys.path.insert(0, patchsae_dir)
import torch
import pandas as pd
import argparse
from torch.utils.data import DataLoader
from collections import OrderedDict
from patchsae.tasks.utils import (
    get_all_classnames,
    get_sae_and_vit,
    load_datasets
)
from Documents.Concept_LoRA.patchsae.integrate_mono import (
    per_class_monosemanticity,
    dataset_monosemanticity,
    topk_neurons_overall
)
import open_clip

def extract_embeddings_and_activations(sae, vit, datasets, device, dataset_name="caltech101"):
    """
    Extract CLIP embeddings and SAE activations using patchsae interface
    """
    
    # Custom collate function to handle PIL images
    def custom_collate(batch):
        images, labels = zip(*batch)
        # Keep images as PIL for individual processing, convert labels to tensor
        return list(images), torch.tensor(labels)
    
    # Create a simple dataset wrapper
    class ProcessedDataset:
        def __init__(self, dataset):
            self.dataset = dataset
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            item = self.dataset[idx]
            return item['image'], item['label']  # Return PIL image and label
    
    dataset = datasets[dataset_name]
    processed_dataset = ProcessedDataset(dataset)
    dataloader = DataLoader(processed_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)
    
    all_embeddings = []
    all_activations = []
    all_labels = []
    
    vit.model.eval()
    sae.eval()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            labels = labels.to(device)
            
            # Process each image individually with CLIP processor
            batch_processed = []
            for img in images:
                processed = vit.processor(images=img, text="", return_tensors="pt", padding=True)
                batch_processed.append(processed)
            
            # Combine batch
            pixel_values = torch.cat([p['pixel_values'] for p in batch_processed], dim=0).to(device)
            input_ids = torch.cat([p['input_ids'] for p in batch_processed], dim=0).to(device)
            attention_mask = torch.cat([p['attention_mask'] for p in batch_processed], dim=0).to(device)
            
            batch_inputs = {
                'pixel_values': pixel_values,
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
            # Get CLIP image embeddings
            image_features = vit.model.get_image_features(pixel_values=pixel_values)
            
            # Get ViT intermediate features using patchsae hooks
            hook_locations = [(sae.cfg.block_layer, sae.cfg.module_name)]
            _, vit_cache_dict = vit.run_with_cache(hook_locations, **batch_inputs)
            vit_features = vit_cache_dict[(sae.cfg.block_layer, sae.cfg.module_name)]
            
            # Get SAE activations
            sae_output, sae_cache_dict = sae.run_with_cache(vit_features)
            sae_activations = sae_cache_dict["hook_hidden_post"]
            
            # Handle SAE activation shape
            if sae_activations.dim() == 3:
                sae_activations = sae_activations[:, 0, :]  # CLS token
            
            all_embeddings.append(image_features.cpu())
            all_activations.append(sae_activations.cpu())
            all_labels.append(labels.cpu())
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(dataloader)}")
    
    image_embeddings = torch.cat(all_embeddings, dim=0)
    neuron_activations = torch.cat(all_activations, dim=0)
    labels = torch.cat(all_labels, dim=0)
    num_classes = len(set(labels.tolist()))
    
    return image_embeddings, neuron_activations, labels, num_classes

def standardize_with_stats(X, mean, std):
    return (X - mean) / std.clamp_min(1e-6)

def convert_openai_to_hf_clip(openai_state_dict):
    """
    Convert full OpenAI CLIP state dict to HuggingFace CLIP format
    Handles both vision and text models
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
            
            # Attention layers
            if '.attn.in_proj_weight' in new_key:
                # Split combined QKV weight
                layer_num = new_key.split('encoder.layers.')[1].split('.')[0]
                base = f'vision_model.encoder.layers.{layer_num}.self_attn.'
                dim = value.shape[0] // 3
                hf_state_dict[base + 'q_proj.weight'] = value[:dim].clone()
                hf_state_dict[base + 'k_proj.weight'] = value[dim:2*dim].clone()
                hf_state_dict[base + 'v_proj.weight'] = value[2*dim:].clone()
                continue
            elif '.attn.in_proj_bias' in new_key:
                # Split combined QKV bias
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
            
            # Vision projection (last layer)
            if new_key == 'vision_model.proj':
                new_key = 'visual_projection.weight'
                value = value.T  # OpenAI uses transposed projection
        
        # Handle text model keys
        elif key.startswith('transformer.') or key in ['token_embedding.weight', 'positional_embedding', 
                                                         'ln_final.weight', 'ln_final.bias', 'text_projection']:
            # Token and position embeddings
            if key == 'token_embedding.weight':
                new_key = 'text_model.embeddings.token_embedding.weight'
            elif key == 'positional_embedding':
                new_key = 'text_model.embeddings.position_embedding.weight'
            
            # Transformer blocks
            elif key.startswith('transformer.resblocks.'):
                new_key = key.replace('transformer.resblocks.', 'text_model.encoder.layers.')
                
                # Attention layers
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
            
            # Final layer norm
            elif key.startswith('ln_final.'):
                new_key = key.replace('ln_final.', 'text_model.final_layer_norm.')
            
            # Text projection
            elif key == 'text_projection':
                new_key = 'text_projection.weight'
                value = value.T  # OpenAI uses transposed projection
        
        # Logit scale (shared parameter)
        elif key == 'logit_scale':
            new_key = 'logit_scale'
        
        # Add to output dict
        if new_key:
            hf_state_dict[new_key] = value
        else:
            print(f"Warning: Unhandled key: {key}")
    
    return hf_state_dict

def analyze_monosemanticity(image_embeddings, neuron_activations, labels, num_classes, save_path=None, dataset_name=""):
    """
    Perform comprehensive monosemantic analysis with CSV output
    """
    print(f"Computing monosemantic scores for {dataset_name}...")
    
    # Overall dataset monosemanticity
    ms_scores, avg_ms = dataset_monosemanticity(image_embeddings, neuron_activations)
    print(f"{dataset_name} - Overall average monosemantic score: {avg_ms:.4f}")
    
    # Top-k neurons overall
    topk_indices, topk_scores = topk_neurons_overall(ms_scores, k=10)
    
    # Calculate AVERAGE of top 10 neurons
    top10_average = topk_scores[:10].mean().item()
    
    print(f"{dataset_name} - Top 10 neurons (individual scores):")
    top10_data = []
    for i, (idx, score) in enumerate(zip(topk_indices[:10], topk_scores[:10])):
        print(f"  {i+1:2d}. Neuron {idx:4d}: {score:.4f}")
        top10_data.append({
            'rank': i+1,
            'neuron_id': idx.item(),
            'monosemantic_score': score.item(),
            'dataset': dataset_name
        })
    
    print(f"{dataset_name} - Average of top 10 neurons: {top10_average:.4f}")
    
    # Add average row to the data
    top10_data.append({
        'rank': 'AVERAGE',
        'neuron_id': 'TOP_10_NEURONS',
        'monosemantic_score': top10_average,
        'dataset': dataset_name
    })
    
    # Create DataFrame and save top 10 CSV
    top10_df = pd.DataFrame(top10_data)
    top10_csv_path = f"{dataset_name}_top10_neurons.csv"
    top10_df.to_csv(top10_csv_path, index=False)
    print(f"Saved top 10 neurons to {top10_csv_path}")
    
    # Per-class monosemanticity
    per_class_results = per_class_monosemanticity(
        image_embeddings, neuron_activations, labels, num_classes, k=10
    )
    
    print(f"\n{dataset_name} - Per-class analysis for {len(per_class_results)} classes:")
    
    # Prepare per-class CSV data
    per_class_data = []
    class_averages = []
    
    for class_id, neuron_scores in per_class_results.items():
        # Calculate average for this class's top 10 neurons
        class_scores = [score for _, score in neuron_scores[:10]]  # Get top 10 scores
        class_average = sum(class_scores) / len(class_scores) if class_scores else 0.0
        class_averages.append(class_average)
        
        print(f"Class {class_id} top neurons (avg: {class_average:.4f}):")
        
        # Add individual neuron data
        for rank, (neuron_idx, score) in enumerate(neuron_scores[:10], 1):
            print(f"  {rank}. Neuron {neuron_idx:4d}: {score:.4f}")
            per_class_data.append({
                'class_id': class_id,
                'rank': rank,
                'neuron_id': neuron_idx,
                'monosemantic_score': score,
                'class_average': class_average,
                'dataset': dataset_name
            })
        
        # Add class average row
        per_class_data.append({
            'class_id': class_id,
            'rank': 'CLASS_AVERAGE',
            'neuron_id': 'TOP_10_FOR_CLASS',
            'monosemantic_score': class_average,
            'class_average': class_average,
            'dataset': dataset_name
        })
    
    # Overall average across all classes
    overall_class_average = sum(class_averages) / len(class_averages) if class_averages else 0.0
    print(f"\n{dataset_name} - Overall average across all classes: {overall_class_average:.4f}")
    
    # Create per-class DataFrame and save CSV
    per_class_df = pd.DataFrame(per_class_data)
    per_class_csv_path = f"{dataset_name}_per_class_top10.csv"
    per_class_df.to_csv(per_class_csv_path, index=False)
    print(f"Saved per-class analysis to {per_class_csv_path}")
    
    # Save comprehensive results if requested
    if save_path:
        results = {
            'overall_scores': ms_scores,
            'average_score': avg_ms,
            'top10_average': top10_average,
            'topk_overall': list(zip(topk_indices.tolist(), topk_scores.tolist())),
            'per_class': per_class_results,
            'per_class_averages': dict(zip(per_class_results.keys(), class_averages)),
            'overall_class_average': overall_class_average
        }
        torch.save(results, f"{dataset_name}_results.pt")
        print(f"Results saved to {dataset_name}_results.pt")
    
    return ms_scores, per_class_results, {
        'top10_average': top10_average,
        'class_averages': class_averages,
        'overall_class_average': overall_class_average
    }

def run_monosemantic_analysis(sae, vit, datasets, device, model_type="baseline"):
    """Helper function to run analysis for a specific model type"""
    print(f"\n=== {model_type.upper()} Model Analysis ===")
    
    results_by_dataset = {}
    
    for dataset_name in datasets.keys():
        print(f"\n--- Analyzing Dataset: {dataset_name.upper()} ({model_type}) ---")
        
        # Extract embeddings and activations
        image_embeddings, neuron_activations, labels, num_classes = extract_embeddings_and_activations(
            sae, vit, datasets, device, dataset_name=dataset_name
        )
        
        print(f"Dataset: {dataset_name} ({model_type})")
        print(f"  Extracted embeddings: {image_embeddings.shape}")
        print(f"  Extracted activations: {neuron_activations.shape}")
        print(f"  Number of classes: {num_classes}")
        
        # Analyze monosemanticity
        ms_scores, per_class_results, averages = analyze_monosemanticity(
            image_embeddings, neuron_activations, labels, num_classes, 
            save_path=f"{dataset_name}_{model_type}_monosemantic_results.pt",
            dataset_name=f"{dataset_name}_{model_type}"
        )

        
        # Store results
        results_by_dataset[dataset_name] = {
            'ms_scores': ms_scores,
            'per_class_results': per_class_results,
            'num_samples': len(labels),
            'num_classes': num_classes,
            'top10_average': averages['top10_average'],
            'overall_class_average': averages['overall_class_average']
        }
        
        print(f"--- Completed {model_type} analysis for {dataset_name} ---")
    
    return results_by_dataset

# def test_model_difference(vit_baseline, vit_lora, datasets, device):
#     """Test if baseline and LoRA models produce different outputs"""
#     print("\n=== TESTING MODEL DIFFERENCE ===")
    
#     # Get a single test image
#     test_dataset = datasets["caltech101"]  
#     test_item = test_dataset[0]
#     test_image = test_item['image']
    
#     # Process the same image through both models
#     processed = vit_baseline.processor(images=test_image, text="", return_tensors="pt", padding=True)
#     processed = {k: v.to(device) for k, v in processed.items()}
    
#     with torch.no_grad():
#         # Get embeddings from baseline
#         baseline_embeddings = vit_baseline.model.get_image_features(pixel_values=processed['pixel_values'])
        
#         # Get embeddings from LoRA  
#         lora_embeddings = vit_lora.model.get_image_features(pixel_values=processed['pixel_values'])
        
#         # Compare
#         diff = (baseline_embeddings - lora_embeddings).abs().mean().item()
#         baseline_norm = baseline_embeddings.norm().item()
#         lora_norm = lora_embeddings.norm().item()
        
#         print(f"Baseline embedding norm: {baseline_norm:.6f}")
#         print(f"LoRA embedding norm: {lora_norm:.6f}")
#         print(f"Absolute difference: {diff:.6f}")
#         print(f"Relative difference: {diff/baseline_norm:.6f}")
        
#         if diff < 1e-6:
#             print("WARNING: Models are producing nearly identical outputs!")
#             return False
#         else:
#             print("Models are producing different outputs")
#             return True



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", type=str, default="ViT-B-16")
    ap.add_argument("--dataset", type=str, default="eurosat", choices=["caltech101", "oxford_pets", "medmnist", "eurosat", "ucf101", "eurosat_extra"], help="Dataset to analyze")
    
    args = ap.parse_args()
    device = args.device
    
    sae_path = "/DATA/cs22btech11053/Concept_Lora/out.pt"
    # sae_path = "/DATA/cs22btech11053/Concept_Lora/Concept_LoRA/models/Concept_lora/clip-vit-base-patch16_-1_resid_49152.pt"
    # sae_path = "/DATA/cs22btech11053/Concept_Lora/Concept_LoRA/models/Concept_lora/clip-vit-base-patch16_-3_resid_49152.pt"

    # checkpoint = torch.load(sae_path, map_location="cpu", weights_only=False)
    # print("Available keys in checkpoint:", list(checkpoint.keys()))
    # if "state_dict" in checkpoint:
    #     print("Available keys in state_dict:", list(checkpoint["state_dict"].keys()))
    # exit(0)


    
    # # Load datasets once
    datasets = load_datasets()
    # print("Datasets loaded:", list(datasets.keys()))
    
    classnames = get_all_classnames(datasets, data_root="/DATA/cs22btech11053/Concept_Lora/Concept_LoRA/patchsae/configs/classnames")
    # print("Classnames loaded for datasets:", {k: len(v) for k, v in classnames.items()})

    
    # # STEP 1: BASELINE ANALYSIS - Load fresh models
    print("\n" + "="*80)
    print("ANALYZING BASELINE (ZERO-SHOT) CLIP MODEL")
    print("="*80)
    
    print("Loading FRESH SAE and ViT for baseline...")
    sae_baseline, vit_baseline, cfg_baseline = get_sae_and_vit(
        sae_path=sae_path, 
        vit_type="base",
        device=device,
        backbone="openai/clip-vit-base-patch16"
    )
    print("Fresh baseline models loaded.")
    
    # Verify we have clean baseline weights
    print("Baseline model info:")
    print(f"  Model ID: {id(vit_baseline.model)}")
    baseline_results = run_monosemantic_analysis(sae_baseline, vit_baseline, datasets, device, model_type="baseline")
    
    # Save baseline results
    baseline_data = []
    for dataset_name in datasets.keys():
        baseline_res = baseline_results[dataset_name]
        baseline_overall = torch.nanmean(baseline_res['ms_scores']).item()
        print(f"\n{dataset_name.upper()} (Baseline):")
        print(f"  Overall avg: {baseline_overall:.4f}, Top10 avg: {baseline_res['top10_average']:.4f}, Class avg: {baseline_res['overall_class_average']:.4f}")
        
        baseline_data.append({
            'dataset': dataset_name,
            'baseline_overall': baseline_overall,
            'baseline_top10': baseline_res['top10_average'],
            'baseline_class_avg': baseline_res['overall_class_average']
        })
    # Save baseline summary CSV
    baseline_df = pd.DataFrame(baseline_data)
    baseline_csv_path = f"baseline_{args.dataset}_summary.csv"
    baseline_df.to_csv(baseline_csv_path, index=False)
    print(f"Baseline results saved to {baseline_csv_path}")
    # exit(0)
    
    
    
    # STEP 2: LORA ANALYSIS - Load fresh models again
    print("\n" + "="*80) 
    print("ANALYZING LORA-TUNED CLIP MODEL")
    print("="*80)
    
    print("Loading FRESH SAE and ViT for LoRA...")
    sae_lora, vit_lora, cfg_lora = get_sae_and_vit(
        sae_path=sae_path,
        vit_type="base", 
        device=device,
        backbone="openai/clip-vit-base-patch16"
    )
    
    
    # NOW load LoRA weights into the fresh model
    merged_weights_path = "/DATA/cs22btech11053/Concept_Lora/Concept_LoRA/clip_lora_weights/clip_vitb16_{}_lora_merged.pt".format(args.dataset)
    state_dict = torch.load(merged_weights_path, map_location=device)
    print("LoRA weights loaded from file.")
    
    # Convert and apply LoRA weights
    hf_state_dict = convert_openai_to_hf_clip(state_dict)
    incomp = vit_lora.model.load_state_dict(hf_state_dict)
    print("LoRA weights applied to fresh model.")
    print(f"  Model ID: {id(vit_lora.model)} (should be different from baseline)")
    print(f"  Missing keys: {len(incomp.missing_keys)}")
    print(f"  Unexpected keys: {len(incomp.unexpected_keys)}")
    
    # Run LoRA analysis
    lora_results = run_monosemantic_analysis(sae_lora, vit_lora, datasets, device, model_type="lora")

    # Similar to baseline, save LoRA summary CSV
    lora_data = []
    for dataset_name in datasets.keys():
        lora_res = lora_results[dataset_name]
        lora_overall = torch.nanmean(lora_res['ms_scores']).item()
        print(f"\n{dataset_name.upper()} (LoRA):")
        print(f"  Overall avg: {lora_overall:.4f}, Top10 avg: {lora_res['top10_average']:.4f}, Class avg: {lora_res['overall_class_average']:.4f}")
        
        lora_data.append({
            'dataset': dataset_name,
            'lora_overall': lora_overall,
            'lora_top10': lora_res['top10_average'],
            'lora_class_avg': lora_res['overall_class_average']
        })
    lora_df = pd.DataFrame(lora_data)
    lora_csv_path = f"lora_{args.dataset}_summary.csv"
    lora_df.to_csv(lora_csv_path, index=False)
    print(f"LoRA results saved to {lora_csv_path}")
    
    # # STEP 3: COMPARISON
    # print("\n" + "="*80)
    # print("BASELINE vs LoRA COMPARISON")
    # print("="*80)
    
    # Add this call in your main function after loading both models:
    # models_are_different = test_model_difference(vit_baseline, vit_lora, datasets, device)
    # if not models_are_different:
    #     print("STOPPING: LoRA weights did not change the model!")
    #     return

if __name__ == "__main__":
    main()
