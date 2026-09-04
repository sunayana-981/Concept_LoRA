import os
import random
from collections import defaultdict

from .utils import Datum, DatasetBase

template = ['a photo of a {}, a type of aircraft.']

_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG')


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


class FGVCAircraft(DatasetBase):
    """FGVC-Aircraft: 100 aircraft variant classes.

    Expected folder structure:
        <root>/fgvc_imagefolder/train/<variant_name>/<image>
    """

    dataset_dir = 'fgvc_imagefolder'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.template = template

        train_dir = os.path.join(self.dataset_dir, 'train')
        classes = sorted(d for d in os.listdir(train_dir)
                         if os.path.isdir(os.path.join(train_dir, d)))

        all_items = []
        for label, cls in enumerate(classes):
            cls_dir = os.path.join(train_dir, cls)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(_IMG_EXTS):
                    all_items.append(Datum(
                        impath=os.path.join(cls_dir, fname),
                        label=label,
                        classname=cls,
                    ))

        train, val, test = _few_shot_split(all_items, num_shots)
        super().__init__(train_x=train, val=val, test=test)
