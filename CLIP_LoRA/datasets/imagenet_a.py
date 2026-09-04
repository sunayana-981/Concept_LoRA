import os
import random
from collections import defaultdict

from .utils import Datum, DatasetBase


def _parse_synset_readme(readme_path):
    """Parse imagenet-a / imagenet-r README to get synset_id -> classname."""
    mapping = {}
    if not os.path.exists(readme_path):
        return mapping
    with open(readme_path) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if (len(parts) == 2
                    and parts[0].startswith('n')
                    and parts[0][1:].isdigit()):
                mapping[parts[0]] = parts[1].strip().replace('_', ' ')
    return mapping


def _load_synset_folder(dataset_dir, synset_to_name):
    """Walk sorted synset subdirectories and collect Datum objects."""
    folders = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and d.startswith('n')
    )
    classnames = [synset_to_name.get(f, f) for f in folders]
    items = []
    for label, folder in enumerate(folders):
        folder_path = os.path.join(dataset_dir, folder)
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                items.append(Datum(
                    impath=os.path.join(folder_path, fname),
                    label=label,
                    classname=classnames[label],
                ))
    return items, classnames


def _few_shot_split(all_items, num_shots):
    """Split all_items into (train, val, test) with N-shot train sampling."""
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
        # Fall back to full class when nothing is left for test
        test.extend(remaining if remaining else items)

    if not val:
        val = train[: max(1, len(train) // 5)]
    return train, val, test


class ImageNetA(DatasetBase):
    """ImageNet-A: 200-class natural adversarial examples.

    Folder structure:
        <root>/imagenet-a/<synset_id>/<image>.jpg
    Class names are read from the README.txt in the dataset root.
    """

    dataset_dir = 'imagenet-a'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        readme = os.path.join(self.dataset_dir, 'README.txt')
        synset_to_name = _parse_synset_readme(readme)

        self.template = ['a photo of a {}.']
        all_items, _ = _load_synset_folder(self.dataset_dir, synset_to_name)
        train, val, test = _few_shot_split(all_items, num_shots)
        super().__init__(train_x=train, val=val, test=test)
