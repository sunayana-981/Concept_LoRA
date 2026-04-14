#!/usr/bin/env python3
"""
Evaluate SAEs with the DAMS formula:

    DAMS = EC × (α × CSS_norm + β × FSS)

    EC        — CKA(X_cls, A_pool) with sigmoid kernel in concept space
    CSS_norm  — Normalised inter-class MMD in SAE concept space
    FSS       — Entropy-based feature specificity (raw activation weights)
"""
import sys, math, os, json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from tasks.utils import load_sae
from dams_metric import compute_dams
import clip as openai_clip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = "/home/sunayana/Documents/Concept_LoRA/lora_weights/vitb16"
DR = "/home/sunayana/Documents/Concept_LoRA/data"
BB = "ViT-B/16"
BS = 64

RUNS = [
    ("Base SAE", "caltech101", "data/sae_weight/base/out.pt", None),
    ("Base SAE", "eurosat",    "data/sae_weight/base/out.pt", None),
    ("Base SAE", "medmnist",   "data/sae_weight/base/out.pt", None),
    ("LoRA SAE", "caltech101",
     "out/checkpoints/caltech101/ted4zuln/final_sparse_autoencoder_openai/clip-vit-base-patch16_-3_resid_49152.pt",
     f"{LR}/caltech101/16shots/seed1/lora_weights.pt"),
    ("LoRA SAE", "eurosat",
     "out/checkpoints/eurosat/bk3rbkcx/final_sparse_autoencoder_openai/clip-vit-base-patch16_-3_resid_49152.pt",
     f"{LR}/eurosat/16shots/seed1/lora_weights.pt"),
    ("LoRA SAE", "medmnist",
     "out/checkpoints/medmnist/d2ygd3bb/final_sparse_autoencoder_openai/clip-vit-base-patch16_-2_resid_49152.pt",
     f"{LR}/medmnist/16shots/seed1/lora_weights.pt"),
]
MED = ["adipose", "background", "debris", "lymphocytes", "mucus",
       "smooth muscle", "normal colon mucosa",
       "cancer-associated stroma", "colorectal adenocarcinoma epithelium"]


def build_clip(lp):
    m, p = openai_clip.load(BB, device=DEVICE)
    if lp is None:
        return m, p
    s = torch.load(lp, map_location=DEVICE, weights_only=False)
    if "weights" not in s:
        m.load_state_dict(s)
        return m, p
    ly, me = s["weights"], s["metadata"]
    sc = me["alpha"] / math.sqrt(me["r"])
    with torch.no_grad():
        for i in range(12):
            ld = ly.get(f"layer_{i+12}", {})
            if not ld:
                continue
            blk = m.visual.transformer.resblocks[i]
            w = blk.attn.in_proj_weight.data
            d = w.shape[1]
            for pr, off in [("q_proj", 0), ("k_proj", d), ("v_proj", 2*d)]:
                try:
                    A = ld[pr]["w_lora_A"] if isinstance(ld.get(pr), dict) else ld[f"{pr}.w_lora_A"]
                    B = ld[pr]["w_lora_B"] if isinstance(ld.get(pr), dict) else ld[f"{pr}.w_lora_B"]
                    w[off:off+d] += (sc * B.float().to(DEVICE) @ A.float().to(DEVICE)).to(w.dtype)
                except Exception:
                    pass
    return m, p


def find_root(path):
    for r, dirs, _ in os.walk(path):
        if dirs:
            sd = os.path.join(r, dirs[0])
            exts = {os.path.splitext(f)[1].lower()
                    for f in os.listdir(sd) if os.path.isfile(os.path.join(sd, f))}
            if exts & {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                return r
    return path


class MedDS(torch.utils.data.Dataset):
    def __init__(self, pp):
        data = np.load(f"{DR}/pathmnist.npz")
        self.imgs = data["test_images"]
        self.labels = data["test_labels"].flatten().astype(int)
        self.pp = pp
        if_ds = datasets.ImageFolder(root=find_root(f"{DR}/pathmnist_imagefolder"))
        im = {n.replace("_", " ").lower(): i for n, i in if_ds.class_to_idx.items()}
        self.lm = {i: im.get(c.lower(), i) for i, c in enumerate(MED)}
        self.nc = len(MED)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return self.pp(Image.fromarray(self.imgs[i])), self.lm[int(self.labels[i])]


class ImgDS(torch.utils.data.Dataset):
    def __init__(self, path, pp, exc=None):
        r = find_root(path)
        full = datasets.ImageFolder(root=r, transform=pp)
        ex = exc or set()
        keep = [i for i, (_, l) in enumerate(full.samples) if full.classes[l] not in ex]
        self._ds = full; self._idx = keep
        kc = sorted({full.classes[full.targets[i]] for i in keep})
        self._m = {full.class_to_idx[c]: ni for ni, c in enumerate(kc)}
        self.nc = len(kc)
    def __len__(self): return len(self._idx)
    def __getitem__(self, i):
        img, l = self._ds[self._idx[i]]
        return img, self._m[l]


class Cap:
    def __init__(self): self.act = None; self._h = None
    def reg(self, blk):
        def f(m, inp, out): self.act = out.detach().float().transpose(0, 1)
        self._h = blk.register_forward_hook(f)
    def rm(self):
        if self._h: self._h.remove()


results = []
loaded = {}

for name, ds_name, sp, lp in RUNS:
    print(f"\n{'='*60}\n{name} / {ds_name}", flush=True)
    sae, cfg = load_sae(sp, DEVICE)
    sae.eval().to(DEVICE)

    if lp not in loaded:
        loaded[lp] = build_clip(lp)
    model, pp = loaded[lp]
    model.eval()

    if ds_name == "medmnist":
        ds = MedDS(pp)
    elif ds_name == "caltech101":
        ds = ImgDS(f"{DR}/caltech-101", pp, exc={"BACKGROUND_Google"})
    else:
        ds = ImgDS(f"{DR}/eurosat/2750", pp)

    loader = DataLoader(ds, batch_size=BS, shuffle=False, num_workers=4, pin_memory=True)
    nl = len(model.visual.transformer.resblocks)
    li = cfg.block_layer if cfg.block_layer >= 0 else nl + cfg.block_layer

    cap = Cap()
    cap.reg(model.visual.transformer.resblocks[li])
    af, al = [], []
    with torch.no_grad():
        for imgs, labs in tqdm(loader, desc="extract", leave=False):
            model.encode_image(imgs.to(DEVICE))
            af.append(cap.act.cpu())
            al.extend(labs.tolist())
    cap.rm()

    feats = torch.cat(af, dim=0)
    nc = ds.nc
    print(f"  feats={feats.shape}, C={nc}", flush=True)

    # Compute full DAMS composite score
    result = compute_dams(
        sae, feats, al, nc,
        device=DEVICE,
        alpha=0.1,        # sweep-optimal: FSS is the most discriminative term
        beta=0.9,
        css_saturation=0.25,
        fss_entropy_sharpening=2.0,
        ec_metric="cka",
        css_metric="mmd",
        fss_method="entropy",
        ec_batch_size=256,
        sae_batch_size=256,
        ec_subsample=2000,
    )
    print(result, flush=True)

    results.append(dict(
        name=name, dataset=ds_name,
        ec=round(result.ec, 4),
        css_raw=round(result.css_raw, 6),
        css_norm=round(result.css_norm, 4),
        fss=round(result.fss, 4),
        dams=round(result.dams, 4),
        mse_d=round(result.recon_mse_per_dim, 6),
    ))

    del sae, feats
    torch.cuda.empty_cache()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n\n" + "="*100)
print(f"{'SAE':<18} {'Dataset':<12} {'EC(CKA)':>9} {'CSS_norm':>10} {'FSS':>8} {'DAMS':>8}")
print("="*100)
for r in results:
    print(f"{r['name']:<18} {r['dataset']:<12} "
          f"{r['ec']:>9.4f} {r['css_norm']:>10.4f} {r['fss']:>8.4f} {r['dams']:>8.4f}")
print("="*100)

os.makedirs("out", exist_ok=True)
with open("out/sae_dams_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved to out/sae_dams_results.json")
