#!/usr/bin/env python3
"""Download and prepare StanfordCars, Flower102, Food101, and SUN397.

This script writes datasets in the folder layout expected by this repo's
dataset loaders and generates split_zhou_*.json files if missing.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import subprocess
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path

# Avoid importing local CLIP_LoRA/datasets package by removing this script's
# directory from sys.path before importing Hugging Face datasets.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() != SCRIPT_DIR]

from datasets import load_dataset
from scipy.io import loadmat


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def wget_resume(url: str, dst: Path) -> None:
    ensure_parent(dst)
    # Always use -c so an interrupted download can resume cleanly.
    run(["wget", "-c", url, "-O", str(dst)])


def extract_tar(archive: Path, dst_dir: Path) -> None:
    print(f"Extracting {archive} -> {dst_dir}", flush=True)
    with tarfile.open(archive, "r:*") as tf:
        tf.extractall(dst_dir)


def extract_zip(archive: Path, dst_dir: Path) -> None:
    print(f"Extracting {archive} -> {dst_dir}", flush=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dst_dir)


def split_trainval(items: list[tuple[str, int, str]], p_val: float = 0.2) -> tuple[list, list]:
    tracker: dict[int, list] = defaultdict(list)
    for item in items:
        tracker[item[1]].append(item)

    train, val = [], []
    for label_items in tracker.values():
        random.shuffle(label_items)
        n_val = round(len(label_items) * p_val)
        if n_val == 0 and len(label_items) > 1:
            n_val = 1
        val.extend(label_items[:n_val])
        train.extend(label_items[n_val:])
    return train, val


def save_split(split_path: Path, train: list, val: list, test: list, path_prefix: Path) -> None:
    path_prefix = path_prefix.resolve()

    def _rel(items: list[tuple[str, int, str]]) -> list[tuple[str, int, str]]:
        out = []
        for impath, label, cname in items:
            rel = str(Path(impath).resolve().relative_to(path_prefix))
            out.append((rel, int(label), cname))
        return out

    data = {"train": _rel(train), "val": _rel(val), "test": _rel(test)}
    with open(split_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] wrote split: {split_path}", flush=True)


def prepare_food101(root: Path, downloads_dir: Path) -> None:
    print("\n=== Food101 ===", flush=True)
    food_dir = root / "Food101"
    split_path = food_dir / "split_zhou_Food101.json"

    if not (food_dir / "images").is_dir():
        archive = downloads_dir / "food-101.tar.gz"
        raw_dir = root / "food-101"
        wget_resume("http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz", archive)
        if not raw_dir.is_dir():
            extract_tar(archive, root)
        if raw_dir.is_dir() and not food_dir.is_dir():
            shutil.move(str(raw_dir), str(food_dir))

    if split_path.is_file():
        print(f"[SKIP] split exists: {split_path}", flush=True)
        return

    image_dir = food_dir / "images"
    meta_dir = food_dir / "meta"
    with open(meta_dir / "train.txt", "r") as f:
        train_ids = [x.strip() for x in f if x.strip()]
    with open(meta_dir / "test.txt", "r") as f:
        test_ids = [x.strip() for x in f if x.strip()]

    classes = sorted({x.split("/")[0] for x in train_ids + test_ids})
    c2l = {c: i for i, c in enumerate(classes)}

    trainval = []
    for rid in train_ids:
        cname = rid.split("/")[0]
        trainval.append((str(image_dir / f"{rid}.jpg"), c2l[cname], cname))

    test = []
    for rid in test_ids:
        cname = rid.split("/")[0]
        test.append((str(image_dir / f"{rid}.jpg"), c2l[cname], cname))

    train, val = split_trainval(trainval, p_val=0.2)
    save_split(split_path, train, val, test, image_dir)


def prepare_flowers102(root: Path, downloads_dir: Path) -> None:
    print("\n=== Flower102 ===", flush=True)
    flower_dir = root / "Flower102"
    image_dir = flower_dir / "jpg"
    split_path = flower_dir / "split_zhou_OxfordFlowers.json"
    label_file = flower_dir / "imagelabels.mat"
    cat_file = flower_dir / "cat_to_name.json"

    flower_dir.mkdir(parents=True, exist_ok=True)
    if not image_dir.is_dir():
        archive = downloads_dir / "102flowers.tgz"
        wget_resume("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz", archive)
        extract_tar(archive, flower_dir)

    if not label_file.is_file():
        wget_resume("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat", label_file)

    if not cat_file.is_file():
        # Zhou split metadata file from CoOp.
        run(
            [
                str(Path.home() / "miniconda3" / "envs" / "dncbm310" / "bin" / "python"),
                "-m",
                "gdown",
                "--fuzzy",
                "https://drive.google.com/file/d/1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0/view?usp=sharing",
                "-O",
                str(cat_file),
            ]
        )

    if split_path.is_file():
        print(f"[SKIP] split exists: {split_path}", flush=True)
        return

    labels = loadmat(label_file)["labels"][0]
    with open(cat_file, "r") as f:
        lab2cname = json.load(f)

    tracker: dict[int, list[str]] = defaultdict(list)
    for i, lab in enumerate(labels, start=1):
        lab = int(lab)
        tracker[lab].append(str(image_dir / f"image_{str(i).zfill(5)}.jpg"))

    train, val, test = [], [], []
    for lab, impaths in tracker.items():
        random.shuffle(impaths)
        n_total = len(impaths)
        n_train = round(n_total * 0.5)
        n_val = round(n_total * 0.2)
        cname = lab2cname.get(str(lab), f"flower_{lab}")
        train.extend([(p, lab - 1, cname) for p in impaths[:n_train]])
        val.extend([(p, lab - 1, cname) for p in impaths[n_train : n_train + n_val]])
        test.extend([(p, lab - 1, cname) for p in impaths[n_train + n_val :]])

    save_split(split_path, train, val, test, image_dir)


def prepare_sun397(root: Path, downloads_dir: Path) -> None:
    print("\n=== SUN397 ===", flush=True)
    sun_dir = root / "SUN397"
    image_dir = sun_dir / "SUN397"
    split_path = sun_dir / "split_zhou_SUN397.json"

    sun_dir.mkdir(parents=True, exist_ok=True)
    if not image_dir.is_dir():
        archive = downloads_dir / "SUN397.tar.gz"
        wget_resume("https://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz", archive)
        # Archive root already contains SUN397/ folder.
        extract_tar(archive, root)

    if not (sun_dir / "Training_01.txt").is_file():
        partitions = downloads_dir / "Partitions.zip"
        wget_resume("https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip", partitions)
        extract_zip(partitions, sun_dir)

    if split_path.is_file():
        print(f"[SKIP] split exists: {split_path}", flush=True)
        return

    with open(sun_dir / "ClassName.txt", "r") as f:
        classnames = [x.strip()[1:] for x in f if x.strip()]
    cname2lab = {c: i for i, c in enumerate(classnames)}

    def _read_split(txt_name: str) -> list[tuple[str, int, str]]:
        out = []
        with open(sun_dir / txt_name, "r") as f:
            lines = [x.strip() for x in f if x.strip()]
        for line in lines:
            imname = line[1:] if line.startswith("/") else line
            raw_class = os.path.dirname(imname)
            label = cname2lab[raw_class]
            parts = raw_class.split("/")[1:]
            cname = " ".join(parts[::-1])
            out.append((str(image_dir / imname), label, cname))
        return out

    trainval = _read_split("Training_01.txt")
    test = _read_split("Testing_01.txt")
    train, val = split_trainval(trainval, p_val=0.2)
    save_split(split_path, train, val, test, image_dir)


def format_car_classname(cname: str) -> str:
    parts = cname.split(" ")
    if len(parts) > 1 and parts[-1].isdigit():
        year = parts.pop(-1)
        parts.insert(0, year)
        return " ".join(parts)
    return cname


def prepare_stanford_cars(root: Path) -> None:
    print("\n=== StanfordCars ===", flush=True)
    cars_dir = root / "StanfordCars"
    cars_train = cars_dir / "cars_train"
    cars_test = cars_dir / "cars_test"
    split_path = cars_dir / "split_zhou_StanfordCars.json"

    cars_dir.mkdir(parents=True, exist_ok=True)

    if not (cars_train.is_dir() and cars_test.is_dir()):
        ds_train = load_dataset("Donghyun99/Stanford-Cars", split="train")
        ds_test = load_dataset("Donghyun99/Stanford-Cars", split="test")
        label_names = ds_train.features["label"].names

        cars_train.mkdir(parents=True, exist_ok=True)
        cars_test.mkdir(parents=True, exist_ok=True)

        trainval = []
        for i, row in enumerate(ds_train):
            impath = cars_train / f"train_{i:06d}.jpg"
            if not impath.is_file():
                row["image"].save(impath, format="JPEG")
            label = int(row["label"])
            cname = format_car_classname(label_names[label])
            trainval.append((str(impath), label, cname))

        test = []
        for i, row in enumerate(ds_test):
            impath = cars_test / f"test_{i:06d}.jpg"
            if not impath.is_file():
                row["image"].save(impath, format="JPEG")
            label = int(row["label"])
            cname = format_car_classname(label_names[label])
            test.append((str(impath), label, cname))
    else:
        trainval, test = None, None

    if split_path.is_file():
        print(f"[SKIP] split exists: {split_path}", flush=True)
        return

    if trainval is None or test is None:
        # Rebuild split metadata from disk if image export already happened.
        ds_train = load_dataset("Donghyun99/Stanford-Cars", split="train")
        ds_test = load_dataset("Donghyun99/Stanford-Cars", split="test")
        label_names = ds_train.features["label"].names

        trainval = []
        for i, row in enumerate(ds_train):
            label = int(row["label"])
            cname = format_car_classname(label_names[label])
            trainval.append((str(cars_train / f"train_{i:06d}.jpg"), label, cname))

        test = []
        for i, row in enumerate(ds_test):
            label = int(row["label"])
            cname = format_car_classname(label_names[label])
            test.append((str(cars_test / f"test_{i:06d}.jpg"), label, cname))

    train, val = split_trainval(trainval, p_val=0.2)
    save_split(split_path, train, val, test, cars_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/home/sunayana/data")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["food101", "flower102", "sun397", "stanford_cars"],
        default=["food101", "flower102", "sun397", "stanford_cars"],
        help="Subset of datasets to download/prepare.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    downloads_dir = root / "_downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    print(f"Dataset root: {root}", flush=True)

    selected = set(args.only)
    if "food101" in selected:
        prepare_food101(root, downloads_dir)
    if "flower102" in selected:
        prepare_flowers102(root, downloads_dir)
    if "sun397" in selected:
        prepare_sun397(root, downloads_dir)
    if "stanford_cars" in selected:
        prepare_stanford_cars(root)

    print("\nAll requested datasets are downloaded/prepared.", flush=True)
    print(f"Use --root_path {root} when running training.", flush=True)


if __name__ == "__main__":
    main()
