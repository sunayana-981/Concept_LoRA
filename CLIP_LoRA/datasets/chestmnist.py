import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


_CLASSES = [
    'atelectasis', 'cardiomegaly', 'effusion', 'infiltration', 'mass',
    'nodule', 'pneumonia', 'pneumothorax', 'consolidation', 'edema',
    'emphysema', 'fibrosis', 'pleural thickening', 'hernia',
]


class _ChestSplit(Dataset):
    def __init__(self, images, labels, indices=None, transform=None):
        self.images = images
        self.labels = labels
        self.indices = np.arange(len(images)) if indices is None else np.asarray(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = int(self.indices[index])
        image = Image.fromarray(self.images[source_index]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        target = torch.as_tensor(self.labels[source_index], dtype=torch.float32)
        return image, target


def _few_shot_multilabel_indices(labels, shots, seed=1):
    """Take up to ``shots`` positive examples for every finding."""
    if shots <= 0:
        return np.arange(len(labels))
    rng = random.Random(seed)
    selected = set()
    for class_index in range(labels.shape[1]):
        positives = np.flatnonzero(labels[:, class_index] > 0).tolist()
        rng.shuffle(positives)
        selected.update(positives[:shots])
    return np.asarray(sorted(selected), dtype=np.int64)


class ChestMNIST:
    """Official ChestMNIST splits from the MedMNIST NPZ release.

    ChestMNIST is a 14-label binary classification task, not a 14-way
    single-label task.  ``multilabel`` tells the unified trainer to use
    BCE-with-logits and mean per-label binary accuracy.
    """

    multilabel = True
    classnames = _CLASSES
    template = [
        'a chest x-ray showing {}.',
        'a medical image showing {}.',
        'a chest radiograph with {}.',
    ]

    def __init__(self, root_path, shots):
        archive = root_path
        if os.path.isdir(archive):
            archive = os.path.join(archive, 'chestmnist.npz')
        if not os.path.isfile(archive):
            raise FileNotFoundError(f'ChestMNIST archive not found: {archive}')

        arrays = np.load(archive)
        train_indices = _few_shot_multilabel_indices(arrays['train_labels'], shots)
        self.train_x = _ChestSplit(
            arrays['train_images'], arrays['train_labels'], train_indices
        )
        # A deterministic 2,048-image validation subset keeps best-checkpoint
        # selection inexpensive; the final metric still uses all 22,433 tests.
        val_count = min(2048, len(arrays['val_images']))
        self.val = _ChestSplit(
            arrays['val_images'], arrays['val_labels'], np.arange(val_count)
        )
        self.test = _ChestSplit(arrays['test_images'], arrays['test_labels'])
