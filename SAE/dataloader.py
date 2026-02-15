# SAE/dataloader.py
import os, json, pickle, torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
from torchvision.transforms import (
    Compose, Resize, CenterCrop, RandomResizedCrop,
    ToTensor, Normalize, ColorJitter, RandomHorizontalFlip,
)

# =========================
# MSCOCO
# =========================
class MSCOCODataset(Dataset):
    def __init__(self, images_dir, annotations_file, subset=1.0, transform=None):
        self.images_dir = images_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        with open(annotations_file, "r") as f:
            data = json.load(f)
        self.images = {img["id"]: img for img in data["images"]}

        subset = float(subset)
        subset = max(0.0, min(1.0, subset))
        self.ids = list(self.images.keys())
        self.ids = self.ids[: int(len(self.ids) * subset)]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), img_id


# =========================
# CUB-200-2011
#   Expect PKLs built once:
#   tools/build_cub_pkls.py → train.pkl, test.pkl in dataset root
# =========================
class CUBLoader(Dataset):
    def __init__(self, root, split="train", transform=None):
        """
        root must contain:
          - images/ (actual images)
          - images.txt, image_class_labels.txt, train_test_split.txt
          - train.pkl, test.pkl (built beforehand)
        """
        self.root = root
        self.split = split
        self.transform = transform or Compose([
            Resize(size=256, interpolation=Image.BILINEAR),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        pkl_path = os.path.join(self.root, f"{split}.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"Missing {pkl_path}. Build PKLs first (see tools/build_cub_pkls.py)."
            )
        self.data = pickle.load(open(pkl_path, "rb"))
        if len(self.data) == 0:
            raise RuntimeError(f"{pkl_path} is empty. Rebuild PKLs and verify CUB txt files.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rec = self.data[idx]
        img_path = rec["img_path"]  # absolute path stored by the PKL builder
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, rec["class_label"]








# =========================
# Public factory
# =========================
def get_dataloader(dataset_name,
                   images_dir,
                   annotations_file,
                   subset=1.0,
                   transform=None,
                   batch_size=32,
                   shuffle=True,
                   num_workers=4):
    """
    dataset_name: "mscoco" | "cifar100" | "cub"
    images_dir:
      - mscoco: path to images folder
      - cifar100: ignored
      - cub: ROOT folder (contains images/ and PKLs)
    annotations_file:
      - mscoco: path to instances_*.json
      - others: ignored
    """
    if dataset_name == "mscoco":
        ds = MSCOCODataset(images_dir, annotations_file, subset=subset, transform=transform)

    elif dataset_name == "cifar100":
        ds = datasets.CIFAR100(root="./datasets/cifar100", train=True, download=True, transform=transform)

    elif dataset_name == "cub":
        ds = CUBLoader(root=images_dir, split="train", transform=transform)

    elif dataset_name == "imagenet":
        ds = datasets.ImageFolder(
        root=os.path.join(images_dir, "train"),
        transform=transform
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if len(ds) == 0:
        raise RuntimeError(f"Empty dataset produced for {dataset_name}. Check paths/PKLs/splits.")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    return ds, loader
