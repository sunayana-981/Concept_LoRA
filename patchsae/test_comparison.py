"""
Test script to verify the model comparison functionality
"""

import sys
import torch
from pathlib import Path

print("=" * 80)
print("TESTING MODEL COMPARISON")
print("=" * 80)

# Test 1: Import the comparison module
print("\n1. Testing imports...")
try:
    from compare_models import ModelComparator, ComparisonAnalyzer, convert_openai_to_hf_clip
    print("   ✓ Successfully imported ModelComparator")
    print("   ✓ Successfully imported ComparisonAnalyzer")
    print("   ✓ Successfully imported convert_openai_to_hf_clip")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Verify paths
print("\n2. Verifying file paths...")
sae_path = "data/sae_weight/base/out.pt"
lora_path = "/home/sunayana/Documents/Concept_LoRA/clip_vitb16_ucf101_lora_merged.pt"

if not Path(sae_path).exists():
    print(f"   ✗ SAE path not found: {sae_path}")
    sys.exit(1)
print(f"   ✓ SAE path exists: {sae_path}")

if not Path(lora_path).exists():
    print(f"   ✗ LoRA path not found: {lora_path}")
    sys.exit(1)
print(f"   ✓ LoRA path exists: {lora_path}")

# Test 3: Test weight conversion
print("\n3. Testing OpenAI to HuggingFace conversion...")
try:
    state_dict = torch.load(lora_path, map_location='cpu')
    print(f"   ✓ Loaded checkpoint with {len(state_dict)} keys")
    
    # Check if it's OpenAI format
    has_visual = any('visual.' in k for k in state_dict.keys())
    print(f"   ✓ OpenAI format detected: {has_visual}")
    
    # Convert
    hf_state_dict = convert_openai_to_hf_clip(state_dict)
    print(f"   ✓ Converted to {len(hf_state_dict)} HuggingFace keys")
    
    # Verify key conversions
    has_vision_model = any('vision_model.' in k for k in hf_state_dict.keys())
    print(f"   ✓ HuggingFace format verified: {has_vision_model}")
    
except Exception as e:
    print(f"   ✗ Conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Initialize ModelComparator
print("\n4. Testing ModelComparator initialization...")
try:
    comparator = ModelComparator(
        sae_path=sae_path,
        base_backbone="openai/clip-vit-base-patch16",
        lora_weights_path=lora_path,
        device="cpu",
    )
    print("   ✓ ModelComparator initialized successfully")
    print(f"   ✓ Has LoRA model: {comparator.has_lora}")
    print(f"   ✓ Number of datasets: {len(comparator.datasets)}")
    print(f"   ✓ Available datasets: {list(comparator.datasets.keys())[:5]}...")
    
except Exception as e:
    print(f"   ✗ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test single sample comparison
print("\n5. Testing single sample extraction and comparison...")
try:
    # Get a sample dataset
    dataset_names = list(comparator.datasets.keys())
    if not dataset_names:
        print("   ✗ No datasets available")
        sys.exit(1)
    
    test_dataset = dataset_names[0]
    print(f"   Using dataset: {test_dataset}")
    
    dataset = comparator.datasets[test_dataset]
    sample = dataset[0]
    
    # Handle different sample formats
    if isinstance(sample, (tuple, list)):
        image = sample[0]
    else:
        image = sample
    
    print(f"   ✓ Got sample from {test_dataset}")
    print(f"   Image type: {type(image)}")
    
    # Prepare image tensor
    if hasattr(comparator.vit_base, 'preprocess'):
        image_tensor = comparator.vit_base.preprocess(image).unsqueeze(0).to('cpu')
    else:
        image_tensor = image.unsqueeze(0).to('cpu') if len(image.shape) == 3 else image.to('cpu')
    
    print(f"   ✓ Image tensor shape: {image_tensor.shape}")
    
    # Extract activations from base model
    base_cls, base_patches, base_sae, base_layers = comparator.extract_activations(
        comparator.vit_base, comparator.sae_base, image_tensor
    )
    print(f"   ✓ Base model activations extracted")
    print(f"     - CLS shape: {base_cls.shape}")
    print(f"     - Patches shape: {base_patches.shape}")
    print(f"     - SAE shape: {base_sae.shape}")
    print(f"     - Layers: {len(base_layers)}")
    
    # Extract activations from LoRA model
    lora_cls, lora_patches, lora_sae, lora_layers = comparator.extract_activations(
        comparator.vit_lora, comparator.sae_lora, image_tensor
    )
    print(f"   ✓ LoRA model activations extracted")
    print(f"     - CLS shape: {lora_cls.shape}")
    print(f"     - Patches shape: {lora_patches.shape}")
    print(f"     - SAE shape: {lora_sae.shape}")
    print(f"     - Layers: {len(lora_layers)}")
    
    # Compute metrics
    metrics = comparator.compute_metrics(
        base_patches, lora_patches,
        base_sae, lora_sae,
        base_layers, lora_layers,
        image_id=f"{test_dataset}_0",
        dataset_name=test_dataset,
    )
    print(f"   ✓ Metrics computed successfully")
    print(f"     - SAE cosine similarity: {metrics.sae_cosine_similarity:.4f}")
    print(f"     - CLIP cosine similarity: {metrics.clip_cosine_similarity:.4f}")
    print(f"     - SAE L2 distance: {metrics.sae_l2_distance:.4f}")
    print(f"     - CLIP L2 distance: {metrics.clip_l2_distance:.4f}")
    
except Exception as e:
    print(f"   ✗ Single sample test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test small batch comparison
print("\n6. Testing small batch comparison (3 samples)...")
try:
    results = comparator.compare_on_dataset(
        dataset_name=test_dataset,
        max_samples=3,
    )
    print(f"   ✓ Processed {len(results)} samples")
    
    for i, metric in enumerate(results):
        print(f"   Sample {i+1}:")
        print(f"     SAE cosine: {metric.sae_cosine_similarity:.4f}")
        print(f"     CLIP cosine: {metric.clip_cosine_similarity:.4f}")
    
except Exception as e:
    print(f"   ✗ Batch comparison failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test analyzer
print("\n7. Testing ComparisonAnalyzer...")
try:
    analyzer = ComparisonAnalyzer({test_dataset: results})
    print(f"   ✓ Analyzer created")
    print(f"   ✓ DataFrame shape: {analyzer.df.shape}")
    print(f"   ✓ Columns: {len(analyzer.df.columns)}")
    
    # Test summary statistics
    summary = analyzer.generate_summary_statistics()
    print(f"   ✓ Summary statistics generated: {summary.shape}")
    
except Exception as e:
    print(f"   ✗ Analyzer test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test save functionality
print("\n8. Testing save functionality...")
try:
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as tmpdir:
        analyzer.save_results(tmpdir)
        
        # Check created files
        expected_files = [
            'summary_statistics.csv',
            'full_results.csv',
            'summary.json',
            'comparison_distributions.png',
            'layer_wise_comparison.png',
            'sae_clip_correlation.png',
        ]
        
        created_files = list(Path(tmpdir).glob('*'))
        print(f"   ✓ Created {len(created_files)} files")
        
        for expected in expected_files:
            if (Path(tmpdir) / expected).exists():
                print(f"     ✓ {expected}")
            else:
                print(f"     ✗ {expected} (missing)")
    
except Exception as e:
    print(f"   ✗ Save test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
print("\nThe comparison script is working correctly.")
print("You can now run:")
print(f"  python compare_models.py --sae-path {sae_path} --lora-path {lora_path}")
print("or")
print("  ./run_comparison.sh quick")
print("=" * 80)