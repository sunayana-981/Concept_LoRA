import os
import random
from collections import defaultdict

from .utils import Datum, DatasetBase, read_json, write_json


template = ['a photo of a {}.']

_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG')


def _scan_imagefolder(split_dir):
    """Return Datum list from <split_dir>/<class_name>/<img> ImageFolder layout."""
    classes = sorted(d for d in os.listdir(split_dir)
                     if os.path.isdir(os.path.join(split_dir, d)))
    items = []
    for label, cls in enumerate(classes):
        cls_dir = os.path.join(split_dir, cls)
        classname = cls.replace('_', ' ')
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(_IMG_EXTS):
                items.append(Datum(
                    impath=os.path.join(cls_dir, fname),
                    label=label,
                    classname=classname,
                ))
    return items


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


class OxfordPets(DatasetBase):
    """Oxford-IIIT Pets: 37 pet-breed classes.

    Expected folder structure:
        <root>/oxford_pets_imagefolder/train/<class>/<image>
    """

    dataset_dir = 'oxford_pets_imagefolder'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.template = template

        all_items = _scan_imagefolder(os.path.join(self.dataset_dir, 'train'))
        train, val, test = _few_shot_split(all_items, num_shots)
        super().__init__(train_x=train, val=val, test=test)

    # ── Static helpers kept for back-compat (used by caltech101, dtd, etc.) ──

    @staticmethod
    def split_trainval(trainval, p_val=0.2):
        p_trn = 1 - p_val
        print(f'Splitting trainval into {p_trn:.0%} train and {p_val:.0%} val')
        tracker = defaultdict(list)
        for idx, item in enumerate(trainval):
            tracker[item.label].append(idx)

        train, val = [], []
        for label, idxs in tracker.items():
            n_val = round(len(idxs) * p_val)
            assert n_val > 0
            random.shuffle(idxs)
            for n, idx in enumerate(idxs):
                item = trainval[idx]
                if n < n_val:
                    val.append(item)
                else:
                    train.append(item)
        return train, val

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath.replace(path_prefix, '')
                if impath.startswith('/'):
                    impath = impath[1:]
                out.append((impath, item.label, item.classname))
            return out

        write_json({'train': _extract(train), 'val': _extract(val), 'test': _extract(test)}, filepath)
        print(f'Saved split to {filepath}')

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            return [
                Datum(impath=os.path.join(path_prefix, impath),
                      label=int(label), classname=classname)
                for impath, label, classname in items
            ]

        print(f'Reading split from {filepath}')
        split = read_json(filepath)
        return _convert(split['train']), _convert(split['val']), _convert(split['test'])
