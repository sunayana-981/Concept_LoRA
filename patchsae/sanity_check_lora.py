#!/usr/bin/env python3
"""
Sanity check: test LoRA accuracy using OpenAI CLIP directly (no HF conversion).
If this gives ~90%, the problem is in the OpenAI→HF conversion.
If this also gives ~10%, the problem is in the LoRA merge or classnames.
"""
import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import clip
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm

LORA_PATH  = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16/medmnist/16shots/seed1/lora_weights.pt"
TRAIN_ROOT = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist_imagefolder"
NPZ_PATH   = "/home/sunayana/Documents/Concept_LoRA/data/pathmnist.npz"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

MEDMNIST_CLASSES = [
    "adipose", "background", "debris", "lymphocytes", "mucus",
    "smooth muscle", "normal colon mucosa",
    "cancer-associated stroma", "colorectal adenocarcinoma epithelium",
]

def get_imagefolder_classnames(root):
    ds = datasets.ImageFolder(root=root)
    idx_to_name = {i: n.replace('_', ' ') for n, i in ds.class_to_idx.items()}
    return [idx_to_name[i] for i in range(len(idx_to_name))]

def build_npz_to_if_mapping(root):
    ds = datasets.ImageFolder(root=root)
    if_map = {n.replace('_', ' ').lower(): i for n, i in ds.class_to_idx.items()}
    return {i: if_map.get(c.lower(), i) for i, c in enumerate(MEDMNIST_CLASSES)}

class NpzTest(Dataset):
    def __init__(self, npz_path, mapping, transform):
        data = np.load(npz_path)
        self.imgs = data['test_images']
        self.labels = data['test_labels'].flatten()
        self.mapping = mapping
        self.transform = transform
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.imgs[i])), self.mapping[int(self.labels[i])]

def extract_lora_AB(ld, proj):
    if proj in ld and isinstance(ld[proj], dict):
        try: return ld[proj]["w_lora_A"], ld[proj]["w_lora_B"]
        except: pass
    try: return ld[f"{proj}.w_lora_A"], ld[f"{proj}.w_lora_B"]
    except: return None, None

def main():
    classnames = get_imagefolder_classnames(TRAIN_ROOT)
    label_map = build_npz_to_if_mapping(TRAIN_ROOT)

    print(f"Classnames (ImageFolder order):")
    for i, c in enumerate(classnames): print(f"  {i}: {c}")

    # Load OpenAI CLIP
    print(f"\nLoading OpenAI CLIP ViT-B/16...")
    model, preprocess = clip.load("ViT-B/16", device=DEVICE)

    # Test BASE accuracy first
    test_ds = NpzTest(NPZ_PATH, label_map, preprocess)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    # Text features
    text_inputs = clip.tokenize([f"a photo of a {c}" for c in classnames]).to(DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Base accuracy
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Base CLIP"):
            img_feat = model.encode_image(imgs.to(DEVICE))
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sims = img_feat @ text_features.t()
            preds = sims.argmax(dim=-1).cpu()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"\n  BASE OpenAI CLIP accuracy: {correct/total*100:.2f}%")

    # Now merge LoRA
    print(f"\nMerging LoRA...")
    lora_state = torch.load(LORA_PATH, map_location=DEVICE)
    layers, meta = lora_state["weights"], lora_state["metadata"]
    scale = meta["alpha"] / meta["r"]
    print(f"  rank={meta['r']}, alpha={meta['alpha']}, scale={scale}")

    with torch.no_grad():
        for i in range(12):
            key = f"layer_{i}"
            if key not in layers: continue
            ld = layers[key]
            w = model.transformer.resblocks[i].attn.in_proj_weight.data
            d = w.shape[1]
            for proj, off in [("q_proj",0),("k_proj",d),("v_proj",2*d)]:
                A, B = extract_lora_AB(ld, proj)
                if A is None: continue
                w[off:off+d] += (scale * (B.float().to(DEVICE) @ A.float().to(DEVICE))).to(w.dtype)

        for i in range(12, 24):
            key = f"layer_{i}"
            if key not in layers: continue
            ld = layers[key]
            w = model.visual.transformer.resblocks[i-12].attn.in_proj_weight.data
            d = w.shape[1]
            for proj, off in [("q_proj",0),("k_proj",d),("v_proj",2*d)]:
                A, B = extract_lora_AB(ld, proj)
                if A is None: continue
                w[off:off+d] += (scale * (B.float().to(DEVICE) @ A.float().to(DEVICE))).to(w.dtype)

    # Recompute text features with LoRA-merged model
    with torch.no_grad():
        text_features_lora = model.encode_text(text_inputs)
        text_features_lora = text_features_lora / text_features_lora.norm(dim=-1, keepdim=True)

    # LoRA accuracy
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="LoRA CLIP"):
            img_feat = model.encode_image(imgs.to(DEVICE))
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sims = img_feat @ text_features_lora.t()
            preds = sims.argmax(dim=-1).cpu()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"\n  LORA OpenAI CLIP accuracy: {correct/total*100:.2f}%")

    # Also test: what if text features DON'T change (use base text features)?
    correct2, total2 = 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="LoRA img + Base text"):
            img_feat = model.encode_image(imgs.to(DEVICE))
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sims = img_feat @ text_features.t()  # base text features
            preds = sims.argmax(dim=-1).cpu()
            correct2 += (preds == labels).sum().item()
            total2 += labels.size(0)
    print(f"  LORA img + BASE text accuracy: {correct2/total2*100:.2f}%")

    print("\nIf LoRA accuracy here is ~90%, the issue is OpenAI→HF conversion.")
    print("If LoRA accuracy here is ~10%, the issue is the LoRA merge itself.")

if __name__ == "__main__":
    main()