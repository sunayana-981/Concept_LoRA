#RA/CLIP_LoRA/datasets/cub2002011.py
import os
import random
from collections import defaultdict

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

class cub2002011:
    """
    Simple CUB-200-2011 dataset wrapper.

    Expected folder layout:
      <root_path>/
        train/    (ImageFolder structure: class subfolders)
        val/
        test/

    Constructor:
      CUB2002011(root_path, shots)

    - shots: if >0, sample 'shots' images per class for train_x (few-shot).
    - Attributes provided by other code: train_x, val, test
    """
    def __init__(self, root_path, shots):
        self.root = root_path
        self.shots = int(shots)

        train_dir = os.path.join(root_path, 'train')
        val_dir = os.path.join(root_path, 'val')
        test_dir = os.path.join(root_path, 'test')

        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"train directory not found: {train_dir}")
        if not os.path.isdir(val_dir):
            raise FileNotFoundError(f"val directory not found: {val_dir}")
        if not os.path.isdir(test_dir):
            raise FileNotFoundError(f"test directory not found: {test_dir}")

        # Use ImageFolder for val/test (will be wrapped by dataloader code with transforms)
        self.train = ImageFolder(train_dir)
        self.val = ImageFolder(val_dir)
        self.test = ImageFolder(test_dir)
        self.num_classes = len(self.train.classes)
        #classnames: human-readable class names used by CLIP prompts
        # folder names in CUB often look like "001.Black_footed_Albatross"
        folders = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        def _pretty(name):
            if '.' in name:
                _, rest = name.split('.', 1)
            else:
                rest = name
            return rest.replace('_', ' ')
        self.classnames = [_pretty(f) for f in folders]

        # default template(s) used by CLIP classifier code
        self.template = ["a photo of a {}."]

        # train_x should be a dataset suitable for training (full or few-shot subset)
        if self.shots > 0:
            self.train_x = self._make_few_shot(self.train, self.shots)
        else:
            self.train_x = self.train

        # train_x should be a dataset suitable for training (full or few-shot subset)
        if self.shots > 0:
            self.train_x = self._make_few_shot(self.train, self.shots)
        else:
            self.train_x = self.train

    def _make_few_shot(self, dataset: ImageFolder, shots_per_class: int):
        """
        Return a Subset of dataset containing shots_per_class samples per class.
        Deterministic selection using a fixed seed.
        """
        # Map class_idx -> list of sample indices
        cls_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset.samples):
            cls_to_indices[label].append(idx)

        selected_indices = []
        rng = random.Random(0)  # deterministic
        for cls, indices in cls_to_indices.items():
            if len(indices) <= shots_per_class:
                # if fewer available than requested, take all
                selected = list(indices)
            else:
                selected = rng.sample(indices, shots_per_class)
            selected_indices.extend(selected)

        # sort indices for stable ordering
        selected_indices = sorted(selected_indices)
        return Subset(dataset, selected_indices)