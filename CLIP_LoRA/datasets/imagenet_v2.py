import os

from .imagenet_a import _few_shot_split
from .imagenet import imagenet_classes
from .utils import Datum, DatasetBase


class ImageNetV2(DatasetBase):
    """ImageNetV2 matched-frequency variant.

    Folder structure:
        <root>/imagenetv2/imagenetv2-matched-frequency-format-val/<int_idx>/<image>.jpeg

    Folders are named 0-999 (integer class indices matching the standard
    ILSVRC label order used in imagenet_classes).
    """

    dataset_dir = 'imagenetv2'
    VARIANT_DIR = 'imagenetv2-matched-frequency-format-val'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        variant_dir = os.path.join(self.dataset_dir, self.VARIANT_DIR)

        # Folders are integer strings; sort numerically
        folders = sorted(
            (d for d in os.listdir(variant_dir)
             if os.path.isdir(os.path.join(variant_dir, d))),
            key=lambda x: int(x),
        )

        self.template = ['a photo of a {}.']

        all_items = []
        for folder in folders:
            label = int(folder)
            folder_path = os.path.join(variant_dir, folder)
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_items.append(Datum(
                        impath=os.path.join(folder_path, fname),
                        label=label,
                        classname=imagenet_classes[label],
                    ))

        train, val, test = _few_shot_split(all_items, num_shots)
        super().__init__(train_x=train, val=val, test=test)
