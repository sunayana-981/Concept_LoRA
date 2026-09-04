import os
import random
from collections import defaultdict

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, Subset

template = ["a photo of a street scene dominated by {}."]


class _RemappedTargetDataset(Dataset):
    """Wrap a dataset and remap class indices to a shared index space."""

    def __init__(self, base_dataset, target_map):
        self.base = base_dataset
        self.target_map = target_map

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, target = self.base[idx]
        return image, self.target_map[target]

    @property
    def transform(self):
        return self.base.transform

    @transform.setter
    def transform(self, value):
        self.base.transform = value

    @property
    def target_transform(self):
        return self.base.target_transform

    @target_transform.setter
    def target_transform(self, value):
        self.base.target_transform = value

    def __getattr__(self, name):
        # Forward optional DatasetFolder attributes used by utilities.
        return getattr(self.base, name)


class Cityscapes:
    """
    Cityscapes scene classification dataset (dominant semantic class per image).

    Expected layout (produced by tasks/convert_cityscapes_to_imagefolder.py):
      <root_path>/cityscapes_imagefolder/
        train/  road/ building/ car/ ...
        val/    road/ building/ car/ ...
    """

    def __init__(self, root_path, shots):
        train_dir = os.path.join(root_path, "cityscapes_imagefolder", "train")
        val_dir = os.path.join(root_path, "cityscapes_imagefolder", "val")
        for d in (train_dir, val_dir):
            if not os.path.isdir(d):
                raise FileNotFoundError(
                    f"Cityscapes imagefolder not found: {d}\n"
                    "Run: python sae_vlm/tasks/convert_cityscapes_to_imagefolder.py"
                )

        self.template = template
        train_full = ImageFolder(train_dir)
        val_full = ImageFolder(val_dir)

        self.classnames = [c.replace("_", " ") for c in train_full.classes]
        train_name_to_idx = train_full.class_to_idx
        train_names = set(train_full.class_to_idx.keys())
        val_names = set(val_full.class_to_idx.keys())

        if train_names != val_names:
            missing_in_val = sorted(train_names - val_names)
            missing_in_train = sorted(val_names - train_names)
            print(
                "[Cityscapes] Class-map sanity check: train/val class sets differ. "
                f"missing_in_val={missing_in_val}, missing_in_train={missing_in_train}"
            )

        # ImageFolder builds indices independently per split. Remap val labels
        # into the train index space so logits/targets are aligned.
        val_target_map = {}
        for val_name, val_idx in val_full.class_to_idx.items():
            if val_name not in train_name_to_idx:
                raise ValueError(
                    f"Validation class '{val_name}' is missing from Cityscapes train split."
                )
            val_target_map[val_idx] = train_name_to_idx[val_name]

        self.val = _RemappedTargetDataset(val_full, val_target_map)
        self.test = self.val  # Cityscapes test labels are withheld; reuse val

        shots = int(shots)
        if shots > 0:
            cls_to_idx = defaultdict(list)
            for idx, (_, label) in enumerate(train_full.samples):
                cls_to_idx[label].append(idx)
            rng = random.Random(0)
            shot_idx = []
            for label in sorted(cls_to_idx):
                pool = cls_to_idx[label]
                shot_idx.extend(rng.sample(pool, min(shots, len(pool))))
            self.train_x = Subset(train_full, sorted(shot_idx))
        else:
            self.train_x = train_full
