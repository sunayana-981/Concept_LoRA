import os
import random
from collections import defaultdict

from .utils import Datum, DatasetBase

_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG')

# PathMNIST 9-class label mapping (label index order from medmnist INFO dict)
_PATHMNIST_CLASSES = [
    'adipose tissue',
    'background',
    'debris',
    'lymphocytes',
    'mucus',
    'smooth muscle',
    'normal colon mucosa',
    'cancer-associated stroma',
    'colorectal adenocarcinoma epithelium',
]

# Map folder names (as they appear on disk) to label indices.
# Folders are named with underscore-separated class names from medmnist labels.
_FOLDER_TO_LABEL = {
    'adipose':                          0,
    'background':                       1,
    'debris':                           2,
    'lymphocytes':                      3,
    'mucus':                            4,
    'smooth_muscle':                    5,
    'normal_colon_mucosa':              6,
    'cancer-associated_stroma':         7,
    'colorectal_adenocarcinoma_epithelium': 8,
}


def _few_shot_split(all_items, num_shots):
    by_class = defaultdict(list)
    for item in all_items:
        by_class[item.label].append(item)

    train, val, test = [], [], []
    for items in by_class.values():
        random.shuffle(items)
        n_tr = min(num_shots, max(1, len(items) - 1))
        n_va = min(4, len(items) - n_tr)
        train.extend(items[:n_tr])
        val.extend(items[n_tr:n_tr + n_va])
        remaining = items[n_tr + n_va:]
        test.extend(remaining if remaining else items)

    if not val:
        val = train[: max(1, len(train) // 5)]
    return train, val, test


class MedMNIST(DatasetBase):
    """PathMNIST loaded from the on-disk imagefolder structure.

    Expected layout:
        <root>/<class_folder>/<image>.png

    All images are stored in a flat imagefolder (no train/val/test subdirs);
    few-shot splits are created at load time.
    """

    def __init__(self, root_path, shots):
        self.template = [
            'a photo of a {}.',
            'a medical image showing {}.',
            'pathology image of {}.',
        ]

        # Collect all images by label
        all_items = []
        folders = sorted(
            d for d in os.listdir(root_path)
            if os.path.isdir(os.path.join(root_path, d))
        )
        for folder in folders:
            if folder not in _FOLDER_TO_LABEL:
                continue
            label = _FOLDER_TO_LABEL[folder]
            classname = _PATHMNIST_CLASSES[label]
            cls_dir = os.path.join(root_path, folder)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(_IMG_EXTS):
                    all_items.append(Datum(
                        impath=os.path.join(cls_dir, fname),
                        label=label,
                        classname=classname,
                    ))

        train, val, test = _few_shot_split(all_items, shots)
        super().__init__(train_x=train, val=val, test=test)

    # Keep classnames as a plain attribute so existing MedMNIST-aware code
    # that checks ds.classnames still works.
    @property
    def classnames(self):
        return _PATHMNIST_CLASSES
