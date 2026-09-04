import os

from .imagenet_a import _few_shot_split
from .imagenet import imagenet_classes
from .utils import Datum, DatasetBase


class ImageNetSketch(DatasetBase):
    """ImageNet-Sketch: 1000-class sketch renditions.

    Folder structure:
        <root>/sketch/<synset_id>/<image>.JPEG

    Folders are named with synset IDs (n01440764, …) sorted alphabetically,
    which matches the standard ILSVRC label order used in imagenet_classes.
    """

    dataset_dir = 'sketch'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)

        # Alphabetically sorted synsets == standard ILSVRC label indices
        folders = sorted(
            d for d in os.listdir(self.dataset_dir)
            if os.path.isdir(os.path.join(self.dataset_dir, d))
        )

        self.template = ['a photo of a {}.']
        # Local list used before DatasetBase sets self._classnames
        _classnames = [imagenet_classes[i] for i in range(len(folders))]

        all_items = []
        for label, folder in enumerate(folders):
            folder_path = os.path.join(self.dataset_dir, folder)
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_items.append(Datum(
                        impath=os.path.join(folder_path, fname),
                        label=label,
                        classname=_classnames[label],
                    ))

        train, val, test = _few_shot_split(all_items, num_shots)
        super().__init__(train_x=train, val=val, test=test)
