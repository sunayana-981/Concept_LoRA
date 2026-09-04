import os
import random
from collections import defaultdict

from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

template = ["a photo of a {}."]


class OfficeHome:
    """
    OfficeHome dataset — merged across all 4 domains (65 classes).

    Expected layout (produced by tasks/setup_officehome_imagefolder.py):
      <root_path>/officehome_imagefolder/
        Alarm_Clock/  Backpack/  ...  (65 class dirs)
    """

    def __init__(self, root_path, shots):
        image_dir = os.path.join(root_path, "officehome_imagefolder")
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(
                f"OfficeHome imagefolder not found: {image_dir}\n"
                "Run: python sae_vlm/tasks/setup_officehome_imagefolder.py"
            )

        self.template = template
        full = ImageFolder(image_dir)
        self.classnames = [c.replace("_", " ") for c in full.classes]

        cls_to_idx = defaultdict(list)
        for idx, (_, label) in enumerate(full.samples):
            cls_to_idx[label].append(idx)

        rng = random.Random(0)

        # fixed 80/10/10 split per class, then few-shot from train
        train_idx, val_idx, test_idx = [], [], []
        for label in sorted(cls_to_idx):
            idxs = cls_to_idx[label][:]
            rng.shuffle(idxs)
            n = len(idxs)
            n_val = max(1, round(n * 0.1))
            n_test = max(1, round(n * 0.1))
            test_idx.extend(idxs[:n_test])
            val_idx.extend(idxs[n_test:n_test + n_val])
            train_idx.extend(idxs[n_test + n_val:])

        self.val = Subset(full, sorted(val_idx))
        self.test = Subset(full, sorted(test_idx))

        shots = int(shots)
        if shots > 0:
            shot_idx = []
            for label in sorted(cls_to_idx):
                pool = [i for i in train_idx if full.targets[i] == label]
                shot_idx.extend(rng.sample(pool, min(shots, len(pool))))
            self.train_x = Subset(full, sorted(shot_idx))
        else:
            self.train_x = Subset(full, sorted(train_idx))
