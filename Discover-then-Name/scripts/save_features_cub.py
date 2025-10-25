import os, math, argparse, torch, numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import OxfordIIITPet, ImageFolder
import clip

def build_dataset(root, preprocess):
    pstruct = os.path.join(root, "cub2002011/images")
    assert os.path.isdir(pstruct), f"cub2002011/images not found at {pstruct}"
    ds = ImageFolder(pstruct, transform=preprocess)
    print(f"Using ImageFolder: {pstruct} (classes={len(ds.classes)})")
    return ds

@torch.no_grad()
def encode_ds(ds, out_dir, device="cuda", batch_size=128, shard_size=10000, train_ratio=0.9, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    model, preprocess = clip.load("ViT-B/16", device=device)
    model.eval()

    # deterministic split
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(ds))
    rng.shuffle(idxs)
    n_train = int(len(ds) * train_ratio)
    train_idx, val_idx = idxs[:n_train], idxs[n_train:]
    subsets = [("train", train_idx), ("train_val", val_idx)]

    print(f"Dataset size={len(ds)} → train={len(train_idx)}, val={len(val_idx)}")
    for split_name, sel in subsets:
        if len(sel) == 0: 
            continue
        loader = DataLoader(Subset(ds, sel), batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
        buf = []
        row_count, shard_id = 0, 0

        # optional: save an index of file paths for traceability
        paths_txt = Path(out_dir) / f"{split_name}_{shard_id}.txt"
        paths_f = open(paths_txt, "w")

        def flush():
            nonlocal buf, shard_id, paths_f
            if not buf: return
            X = torch.cat(buf, dim=0).half().cpu()       # [M,512], fp16
            out_pt = Path(out_dir) / f"{split_name}_{shard_id}.pt"
            torch.save(X, out_pt)
            paths_f.close()
            print(f"wrote {out_pt}  ({X.shape[0]} rows)")
            shard_id += 1
            buf = []
            paths_f = open(Path(out_dir) / f"{split_name}_{shard_id}.txt", "w")

        # need raw paths; both datasets expose .imgs as [(path, label), ...]
        # Subset wraps it, but order matches sel
        base_imgs = getattr(ds, "imgs", None)
        for b, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)
            feats = model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)   # cosine space
            buf.append(feats.float().cpu())                    # keep float32 in RAM, save fp16
            # write paths for this batch
            start = b * loader.batch_size
            end = start + imgs.size(0)
            for i in range(start, end):
                # resolve original index
                orig_idx = int(sel[i])
                path = ds.imgs[orig_idx][0] if base_imgs is not None else ds.samples[orig_idx][0]
                paths_f.write(path + "\n")

            row_count += imgs.size(0)
            if row_count >= shard_size:
                flush()
                row_count = 0

        flush()  # last partial shard

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/sunayana/Documents/Concept_LoRA/datasets/")
    ap.add_argument("--out",  default="/home/sunayana/Documents/Concept_LoRA/datasets/cub_features/")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--shard_size", type=int, default=10000)
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print("Encoding Oxford Pets dataset features with CLIP ViT-B/16...")

    # just to instantiate preprocess for build_dataset (we won’t use it after)
    _, preprocess = clip.load("ViT-B/16", device="cpu")
    ds = build_dataset(args.root, preprocess)
    encode_ds(ds, args.out, device=args.device, batch_size=args.batch_size,
              shard_size=args.shard_size, train_ratio=args.train_ratio, seed=args.seed)
