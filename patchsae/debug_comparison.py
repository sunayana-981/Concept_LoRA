"""
Debug script to identify issues with compare_models.py
"""

import sys
import traceback

print("=" * 80)
print("DEBUGGING COMPARISON SCRIPT")
print("=" * 80)

# Step 1: Check Python version
print("\n1. Python Version:")
print(f"   {sys.version}")

# Step 2: Check basic imports
print("\n2. Checking basic imports...")
try:
    import torch
    print(f"   ✓ torch {torch.__version__}")
except Exception as e:
    print(f"   ✗ torch: {e}")

try:
    import numpy as np
    print(f"   ✓ numpy {np.__version__}")
except Exception as e:
    print(f"   ✗ numpy: {e}")

try:
    import pandas as pd
    print(f"   ✓ pandas {pd.__version__}")
except Exception as e:
    print(f"   ✗ pandas: {e}")

try:
    import matplotlib
    print(f"   ✓ matplotlib {matplotlib.__version__}")
except Exception as e:
    print(f"   ✗ matplotlib: {e}")

try:
    from transformers import CLIPModel
    print(f"   ✓ transformers (CLIPModel available)")
except Exception as e:
    print(f"   ✗ transformers: {e}")

# Step 3: Check project-specific imports
print("\n3. Checking project-specific imports...")
try:
    from src.demo.core import SAETester
    print(f"   ✓ src.demo.core.SAETester")
except Exception as e:
    print(f"   ✗ src.demo.core.SAETester: {e}")
    traceback.print_exc()

try:
    from tasks.utils import (
        get_all_classnames,
        get_max_acts_and_images,
        get_sae_and_vit,
        load_datasets,
    )
    print(f"   ✓ tasks.utils (all functions)")
except Exception as e:
    print(f"   ✗ tasks.utils: {e}")
    traceback.print_exc()

# Step 4: Check file paths
print("\n4. Checking file paths...")
import os

paths_to_check = [
    ("SAE path", "data/sae_weight/base/out.pt"),
    ("LoRA path", "/home/sunayana/Documents/Concept_LoRA/clip_vitb16_ucf101_lora_merged.pt"),
    ("src.demo.core", "src/demo/core.py"),
    ("tasks.utils", "tasks/utils.py"),
]

for name, path in paths_to_check:
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"   {status} {name}: {path}")
    if not exists:
        print(f"      File/directory not found!")

# Step 5: Check current working directory
print("\n5. Current working directory:")
print(f"   {os.getcwd()}")

# Step 6: Check Python path
print("\n6. Python path:")
for p in sys.path[:5]:
    print(f"   - {p}")

# Step 7: Try to import the comparison module
print("\n7. Attempting to import compare_models...")
try:
    import compare_models
    print("   ✓ compare_models imported successfully")
    print(f"   Available classes: {[name for name in dir(compare_models) if not name.startswith('_')]}")
except Exception as e:
    print(f"   ✗ Failed to import compare_models: {e}")
    traceback.print_exc()

# Step 8: Check if running from correct directory
print("\n8. Directory structure check...")
expected_dirs = ['src', 'tasks', 'data', 'configs']
for d in expected_dirs:
    exists = os.path.exists(d)
    status = "✓" if exists else "✗"
    print(f"   {status} {d}/")

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)