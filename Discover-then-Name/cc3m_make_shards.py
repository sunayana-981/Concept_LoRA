import os, csv, math, argparse
from PIL import Image
from io import BytesIO
import webdataset as wds

def read_pairs(tsv_path, img_root):
    pairs = []
    with open(tsv_path, "r") as f:
        r = csv.reader(f, delimiter="\t")
        for row in r:
            if not row: continue
            rel = row[0]; caption = row[1] if len(row) > 1 else ""
            img_path = rel if os.path.isabs(rel) else os.path.join(img_root, rel)
            pairs.append((img_path, caption))
    return pairs

def img_to_jpg_bytes(path):
    with Image.open(path).convert("RGB") as im:
        buf = BytesIO(); im.save(buf, format="JPEG", quality=95)
    return buf.getvalue()

def write_shards(pairs, out_dir, prefix, max_per_shard=20000):
    os.makedirs(out_dir, exist_ok=True)
    nshards = math.ceil(len(pairs)/max_per_shard) or 1
    pattern = os.path.join(out_dir, f"{prefix}-%05d.tar")
    with wds.ShardWriter(pattern, maxcount=max_per_shard) as sink:
        for i,(p,cap) in enumerate(pairs):
            try:
                jpg = img_to_jpg_bytes(p)
            except Exception:
                continue
            key = f"{prefix}-{i:09d}"
            sink.write({"__key__": key, "jpg": jpg, "txt": cap})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_tsv", required=True)
    ap.add_argument("--val_tsv", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--out_root", default="./data/CC3M_TAR")
    ap.add_argument("--max_per_shard", type=int, default=20000)
    args = ap.parse_args()

    train = read_pairs(args.train_tsv, args.img_root)
    val   = read_pairs(args.val_tsv,   args.img_root)

    write_shards(train, os.path.join(args.out_root,"training"), "train", args.max_per_shard)
    write_shards(val,   os.path.join(args.out_root,"validation"), "val",   args.max_per_shard)
    print("Done. Shards at:", args.out_root)
