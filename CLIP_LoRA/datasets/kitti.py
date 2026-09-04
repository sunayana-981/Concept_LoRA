import os
import random
from collections import defaultdict

from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

KITTI_CLASSES = [
    "Car", "Van", "Truck", "Pedestrian",
    "Person_sitting", "Cyclist", "Tram",
]

template = ["a photo of a {} on the road."]


class KITTI:
    """
    KITTI scene classification dataset.

    Expected layout (produced by tasks/convert_kitti_to_imagefolder.py):
      <root_path>/kitti_imagefolder/
        train/  Car/ Van/ Truck/ ...
        test/   Car/ Van/ Truck/ ...
    """

    def __init__(self, root_path, shots):
        train_dir = os.path.join(root_path, "kitti_imagefolder", "train")
        test_dir = os.path.join(root_path, "kitti_imagefolder", "test")
        for d in (train_dir, test_dir):
            if not os.path.isdir(d):
                raise FileNotFoundError(
                    f"KITTI imagefolder not found: {d}\n"
                    "Run: python sae_vlm/tasks/convert_kitti_to_imagefolder.py"
                )

        self.template = template
        train_full = ImageFolder(train_dir)
        test_full = ImageFolder(test_dir)

        self.classnames = [c.replace("_", " ") for c in train_full.classes]

        # carve a val set from train (10% per class)
        cls_to_idx = defaultdict(list)
        for idx, (_, label) in enumerate(train_full.samples):
            cls_to_idx[label].append(idx)

        rng = random.Random(0)
        train_idx, val_idx = [], []
        for label in sorted(cls_to_idx):
            idxs = cls_to_idx[label][:]
            rng.shuffle(idxs)
            n_val = max(1, round(len(idxs) * 0.1))
            val_idx.extend(idxs[:n_val])
            train_idx.extend(idxs[n_val:])

        self.val = Subset(train_full, sorted(val_idx))
        self.test = test_full

        shots = int(shots)
        if shots > 0:
            shot_idx = []
            for label in sorted(cls_to_idx):
                pool = [i for i in train_idx if train_full.targets[i] == label]
                shot_idx.extend(rng.sample(pool, min(shots, len(pool))))
            self.train_x = Subset(train_full, sorted(shot_idx))
        else:
            self.train_x = Subset(train_full, sorted(train_idx))
