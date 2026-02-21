"""
Convert PathMNIST to imagefolder format compatible with HuggingFace load_dataset.

This saves PathMNIST images into a class-folder structure:
    output_dir/
        adipose/
            train_00000.png
            train_00001.png
            ...
        background/
            ...
        ...

Then you can load it with:
    load_dataset("imagefolder", data_dir="output_dir", split="train")

Usage:
    python convert_pathmnist_to_imagefolder.py \
        --medmnist_root /home/sunayana/Documents/Concept_LoRA/data \
        --output_dir /home/sunayana/Documents/Concept_LoRA/data/pathmnist_imagefolder \
        --split train
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm

try:
    import medmnist
    from medmnist import INFO
except ImportError:
    print("[FATAL] medmnist not installed. Run: pip install medmnist")
    exit(1)


def main():
    parser = argparse.ArgumentParser(description="Convert PathMNIST to imagefolder")
    parser.add_argument(
        "--medmnist_root", type=str,
        default="/home/sunayana/Documents/Concept_LoRA/data",
        help="Root directory where medmnist downloads data"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/home/sunayana/Documents/Concept_LoRA/data/pathmnist_imagefolder",
        help="Output directory for imagefolder format"
    )
    parser.add_argument(
        "--split", type=str, default="train",
        choices=["train", "val", "test", "all"],
        help="Which split(s) to convert. 'all' combines train+val+test."
    )
    args = parser.parse_args()

    data_flag = "pathmnist"
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])

    # Get classnames
    classnames = {int(k): v for k, v in info["label"].items()}
    print(f"[INFO] PathMNIST classes ({len(classnames)}):")
    for idx, name in classnames.items():
        print(f"  {idx}: {name}")

    # Determine which splits to process
    if args.split == "all":
        splits = ["train", "val", "test"]
    else:
        splits = [args.split]

    output_dir = Path(args.output_dir)
    total_saved = 0

    for split in splits:
        print(f"\n[INFO] Processing split: {split}")
        dataset = DataClass(split=split, download=True, root=args.medmnist_root)

        for idx in tqdm(range(len(dataset)), desc=f"  Saving {split}"):
            image, label = dataset[idx]

            # Flatten label
            if hasattr(label, "__len__"):
                label_int = int(label[0])
            else:
                label_int = int(label)

            # Sanitize classname for folder name (replace spaces, slashes)
            classname = classnames[label_int]
            safe_classname = classname.replace("/", "_").replace(" ", "_").replace(",", "")

            # Create class directory
            class_dir = output_dir / safe_classname
            class_dir.mkdir(parents=True, exist_ok=True)

            # Save image
            filename = f"{split}_{idx:06d}.png"
            image.save(class_dir / filename)
            total_saved += 1

    print(f"\n[SUCCESS] Saved {total_saved} images to {output_dir}")
    print(f"\nDirectory structure:")
    for d in sorted(output_dir.iterdir()):
        if d.is_dir():
            n_files = len(list(d.glob("*.png")))
            print(f"  {d.name}/  ({n_files} images)")

    # Print the DATASET_INFO entry to add
    print(f"""
============================================================
Add this to DATASET_INFO in tasks/utils.py:
============================================================

    "medmnist": {{
        "path": "imagefolder",
        "data_dir": "{output_dir}",
        "split": "train",
        "trust_remote_code": True,
    }},

============================================================
And create a classnames file at:
    configs/classnames/medmnist_classnames.json

with contents:
{list(classnames.values())}
============================================================
""")

    # Auto-create the classnames JSON
    import json
    classnames_dir = Path("configs/classnames")
    classnames_dir.mkdir(parents=True, exist_ok=True)
    classnames_file = classnames_dir / "medmnist_classnames.json"
    with open(classnames_file, "w") as f:
        json.dump(list(classnames.values()), f, indent=2)
    print(f"[INFO] Classnames JSON saved to: {classnames_file}")


if __name__ == "__main__":
    main()