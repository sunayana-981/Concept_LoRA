import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from src.demo.core import SAETester
from tasks.utils import (
    get_all_classnames,
    get_max_acts_and_images,
    get_sae_and_vit,
    load_datasets,
)

# Configuration
root = "out/feature_data"
sae_runname = "sae_base"
vit_name = "base"
out_dir = "out/csv"
device =  "cpu"
os.makedirs(out_dir, exist_ok=True)

print(f"Using device: {device}")

# 1. Initialize Models
sae_path = "data/sae_weight/base/out.pt"
sae, vit, cfg = get_sae_and_vit(
    sae_path=sae_path,
    vit_type="base",
    device=device,
    backbone="openai/clip-vit-base-patch16",
    model_path=None,
    classnames=None,
)

# 2. Move to device and set to eval mode
vit.to(device)
vit.eval()
sae.to(device).eval()

# 3. Load metadata and data
datasets = load_datasets() 
max_act_imgs, mean_acts = get_max_acts_and_images(datasets, root, sae_runname, vit_name)
classnames = get_all_classnames(datasets, data_root="configs/classnames")

# 4. Initialize Tester
sae_clip = SAETester(vit, cfg, sae, mean_acts, max_act_imgs, datasets, classnames)

# Optional limit per dataset (None = full dataset)
num_limit = None

# 5. Processing Loop
for ds_name, ds in datasets.items():
    rows = []
    try:
        total = len(ds)
    except Exception:
        total = None

    print(f"\nProcessing dataset: {ds_name} (total={total})")
    iterator = enumerate(ds)
    if total is not None:
        iterator = tqdm(iterator, total=total, desc=ds_name)

    for i, item in iterator:
        if (num_limit is not None) and (i >= num_limit):
            break

        # Get image
        img = item.get("image") or item.get("jpg")
        
        # --- FIX: Ensure data is on the same device as the model ---
        if isinstance(img, torch.Tensor):
            img = img.to(device)
        # -----------------------------------------------------------

        try:
            # Register image (preprocesses and runs forward pass)
            sae_clip.register_image(img)

            # Get activation distribution
            # Note: Ensure get_activation_distribution returns a CPU numpy array or tensor
            filtered = sae_clip.get_activation_distribution()
            
            # If filtered is a torch tensor, move to cpu for numpy ops
            if torch.is_tensor(filtered):
                per_feature = filtered.sum(0).detach().cpu().numpy()
            else:
                per_feature = filtered.sum(0)

            # Top 10 neurons
            top_neurons = np.argsort(per_feature)[::-1][:10]

            # Store row-wise
            for rank, idx in enumerate(top_neurons):
                rows.append({
                    "dataset": ds_name,
                    "image_idx": i,
                    "rank": rank,
                    "neuron_idx": int(idx),
                    "activation": float(per_feature[idx]),
                })
        except Exception as e:
            print(f"Error processing image {i} in {ds_name}: {e}")
            continue

    # Save CSV per dataset
    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(out_dir, f"{ds_name}_top_neurons.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")
    else:
        print(f"No data collected for {ds_name}")

print("\nProcessing Complete.")