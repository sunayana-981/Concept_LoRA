import json
import os
import random
from collections import defaultdict

from .utils import Datum, DatasetBase

template = ['a photo of a {}.']

_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG')

# Oxford 102 Flowers names keyed by the original 1-based class ID.
#
# The local image folders were converted from nelorth/oxford-flowers parquet.
# Its ClassLabel vocabulary is the *lexicographic* ordering of string IDs:
#   0 -> "1", 1 -> "10", 2 -> "100", ..., 14 -> "2", ...
# Therefore folder i is an encoded ClassLabel index, not original_id - 1.
_CAT_TO_NAME_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'sae_vlm', 'configs', 'classnames', 'oxford_flowers_classnames.json',
)


def _load_flower_names():
    try:
        with open(_CAT_TO_NAME_PATH) as f:
            d = json.load(f)
        encoded_ids = sorted(d, key=str)
        return [d[original_id] for original_id in encoded_ids]
    except Exception:
        return [str(i) for i in range(102)]


_FLOWER_NAMES = _load_flower_names()


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


class OxfordFlowers(DatasetBase):
    """Oxford 102 Flowers.

    Expected folder structure:
        <root>/flowers102_imagefolder/train/<int>/<image>
        <root>/flowers102_imagefolder/test/<int>/<image>

    Folder names are 0-based encoded labels from the source parquet's
    lexicographically ordered ClassLabel vocabulary.
    """

    dataset_dir = 'flowers102_imagefolder'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.template = template

        train_dir = os.path.join(self.dataset_dir, 'train')
        test_dir = os.path.join(self.dataset_dir, 'test')

        train_all = self._scan(train_dir)
        test_items = self._scan(test_dir)

        train, val, _ = _few_shot_split(train_all, num_shots)
        super().__init__(train_x=train, val=val, test=test_items)

    @staticmethod
    def _scan(split_dir):
        """Walk integer-named class folders; map index i → _FLOWER_NAMES[i]."""
        folders = sorted(
            (d for d in os.listdir(split_dir)
             if os.path.isdir(os.path.join(split_dir, d))),
            key=lambda x: int(x),
        )
        items = []
        for folder in folders:
            label = int(folder)
            classname = _FLOWER_NAMES[label] if label < len(_FLOWER_NAMES) else str(label)
            cls_dir = os.path.join(split_dir, folder)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(_IMG_EXTS):
                    items.append(Datum(
                        impath=os.path.join(cls_dir, fname),
                        label=label,
                        classname=classname,
                    ))
        return items
