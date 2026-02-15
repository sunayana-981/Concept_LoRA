import argparse, os, sys, random
from pathlib import Path
from PIL import Image
import numpy as np

# your corruption defs (the file that contains get_corruptions)
from generate_corruption import get_corruptions

# ---------- helpers ----------
def iter_images(root, exts=(".jpg", ".jpeg", ".png")):
    root = Path(root)
    for ext in exts:
        for p in root.rglob(f"*{ext}"):
            # skip already-generated CUB-C (avoid recursive writes)
            if "severity_" in str(p) or any(k in str(p) for k in ("gaussian_noise","jpeg_compression","pixelate")):
                continue
            yield p

def ensure_rgb(pil_im):
    return pil_im.convert("RGB") if pil_im.mode != "RGB" else pil_im

def save_numpy_img(x_np, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x_np = np.asarray(x_np)
    if x_np.dtype != np.uint8:
        x_np = np.uint8(np.clip(x_np, 0, 255))
    if x_np.ndim == 2:
        x_np = np.repeat(x_np[..., None], 3, axis=2)
    Image.fromarray(x_np).save(out_path, quality=95)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cub_root", required=True, help="Path to CUB (folder that contains train/val/test or images/)")
    ap.add_argument("--out_root", required=True, help="Output root (CUB-C)")
    ap.add_argument("--severities", type=int, nargs="+", default=[1,2,3,4,5])
    ap.add_argument("--subset", type=float, default=1.0, help="0-1 fraction of images to process")
    ap.add_argument("--corruptions", type=str, nargs="*", default=None, help="Subset of corruption names")
    ap.add_argument("--dry_run", action="store_true", help="List what would be processed and exit")
    args = ap.parse_args()

    corrs = get_corruptions()
    if args.corruptions is not None:
        bad = [c for c in args.corruptions if c not in corrs]
        if bad:
            raise ValueError(f"Unknown corruptions: {bad}\nAvailable: {list(corrs.keys())}")
        corrs = {k: corrs[k] for k in args.corruptions}

    paths = list(iter_images(args.cub_root))
    if len(paths) == 0:
        print("[ERR] No images found under", args.cub_root, file=sys.stderr)
        sys.exit(1)

    if args.subset < 1.0:
        random.seed(123)
        k = max(1, int(len(paths) * args.subset))
        paths = random.sample(paths, k)

    if args.dry_run:
        print(f"[DRY] {len(paths)} images, corrs={list(corrs.keys())}, severities={args.severities}")
        for p in paths[:10]:
            print(" -", p)
        return

    # Some corruptions need extras; skip cleanly if missing
    frost_needed = (Path("frost_images").exists() and any(Path("frost_images").glob("frost*.*")))
    pil_corruptions = {"jpeg_compression", "pixelate"}  # these return PIL

    out_root = Path(args.out_root)
    cub_root = Path(args.cub_root)

    for img_path in paths:
        rel = img_path.relative_to(cub_root)  # preserves train/val/test layout (or images/ if present)
        try:
            pil_im = ensure_rgb(Image.open(img_path))
        except Exception as e:
            print(f"[WARN] open {img_path}: {e}", file=sys.stderr)
            continue

        for cname, cfunc in corrs.items():
            if cname == "frost" and not frost_needed:
                # skip frost if overlays not present
                continue

            for sev in args.severities:
                out_path = out_root / cname / f"severity_{sev}" / rel.with_suffix(".jpg")
                try:
                    if cname in pil_corruptions:
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        ensure_rgb(cfunc(pil_im.copy(), severity=sev)).save(out_path, quality=95)
                    else:
                        x_np = cfunc(pil_im.copy(), severity=sev)  # expects PIL, returns np
                        save_numpy_img(x_np, out_path)
                except Exception as e:
                    print(f"[WARN] {img_path.name} -> {cname}@{sev}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
