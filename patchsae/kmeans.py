import os
import shutil
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PIL import Image

from src.demo.core import SAETester
from tasks.utils import (
    get_all_classnames,
    get_max_acts_and_images,
    get_sae_and_vit,
    load_datasets,
)

# --- Configuration ---
root = "out/feature_data"
sae_runname = "sae_base"
vit_name = "base"
out_base_dir = "out/clustering"
device = "cuda" if torch.cuda.is_available() else "cpu"

num_clusters = 10 
sample_limit = 2000  # Capping images per dataset for t-SNE interpretability
images_to_save_per_cluster = 20

os.makedirs(out_base_dir, exist_ok=True)

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




vit.to(device)
vit.eval()
sae.to(device).eval()

datasets = load_datasets() 
max_act_imgs, mean_acts = get_max_acts_and_images(datasets, root, sae_runname, vit_name)
classnames = get_all_classnames(datasets, data_root="configs/classnames")

sae_clip = SAETester(vit, cfg, sae, mean_acts, max_act_imgs, datasets, classnames, device=device)

# 2. Main Loop
for ds_name, ds in datasets.items():
    all_activations = []
    collected_images = []
    
    print(f"\nProcessing Dataset: {ds_name}")
    
    # Extraction
    count = 0
    for i, item in enumerate(tqdm(ds, total=min(len(ds), sample_limit))):
        if count >= sample_limit:
            break

        img = item.get("image") or item.get("jpg")
        if img is None:
            continue
            
        try:
            sae_clip.register_image(img)
            act = sae_clip.get_activation_distribution() # [tokens, features]
            img_vector = act.mean(axis=0) 
            all_activations.append(img_vector)
            collected_images.append(img)
            count += 1
        except Exception as e:
            continue

    if not all_activations:
        continue

    X = np.array(all_activations)
    
    # 3. Clustering & Dimensionality Reduction
    print(f"Normalizing and reducing {ds_name}...")
    X_scaled = StandardScaler().fit_transform(X)
    
    # PCA to 50 components first (Standard practice for t-SNE)
    pca_pre = PCA(n_components=min(50, X.shape[0], X.shape[1]))
    X_pca = pca_pre.fit_transform(X_scaled)
    
    print(f"Running K-Means (k={num_clusters})...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca)

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_pca)

    # 4. Plotting t-SNE
    plt.figure(figsize=(12, 8))
    for cluster_id in range(num_clusters):
        points = X_tsne[cluster_labels == cluster_id]
        plt.scatter(points[:, 0], points[:, 1], s=20, label=f'Cluster {cluster_id}', alpha=0.6)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f"t-SNE Projection of SAE Clusters: {ds_name}")
    plt.tight_layout()
    
    plot_path = os.path.join(out_base_dir, f"{ds_name}_tsne.png")
    plt.savefig(plot_path)
    plt.close()

    # 5. Image Saving per Cluster
    print(f"Saving representative images for {ds_name}...")
    ds_dir = os.path.join(out_base_dir, ds_name)
    if os.path.exists(ds_dir):
        shutil.rmtree(ds_dir) # Clear old results
    os.makedirs(ds_dir, exist_ok=True)

    for cluster_id in range(num_clusters):
        cluster_path = os.path.join(ds_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_path, exist_ok=True)
        
        # Get indices of images in this cluster
        indices = np.where(cluster_labels == cluster_id)[0]
        
        for rank, idx in enumerate(indices[:images_to_save_per_cluster]):
            img = collected_images[idx]
            
            # Ensure PIL format
            if torch.is_tensor(img):
                from torchvision.transforms import ToPILImage
                img = ToPILImage()(img.detach().cpu())
            elif not isinstance(img, Image.Image):
                img = Image.fromarray(np.uint8(img))
                
            img.save(os.path.join(cluster_path, f"rank_{rank}_idx_{idx}.png"))

    # 6. Interpretability (Top Neurons per Cluster)
    cluster_summary = []
    # Note: Using original X for centroid interpretation as PCA dims are hard to read
    for i in range(num_clusters):
        member_acts = X[cluster_labels == i]
        centroid = member_acts.mean(axis=0)
        top_neurons = np.argsort(centroid)[::-1][:5]
        top_values = centroid[top_neurons]
        
        cluster_summary.append({
            "cluster": i,
            "top_neurons": top_neurons.tolist(),
            "top_neuron_activations": top_values.tolist()
        })

    summary_df = pd.DataFrame(cluster_summary)
    summary_df.to_csv(os.path.join(out_base_dir, f"{ds_name}_cluster_logic.csv"), index=False)

print("\nAll datasets processed successfully.")