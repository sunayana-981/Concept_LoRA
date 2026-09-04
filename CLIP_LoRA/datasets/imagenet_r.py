import os

from .imagenet_a import _parse_synset_readme, _load_synset_folder, _few_shot_split
from .utils import DatasetBase


class ImageNetR(DatasetBase):
    """ImageNet-R: 200-class artistic renditions (art, cartoons, sketches, etc.).

    Folder structure:
        <root>/imagenet-r/<synset_id>/<image>.jpg
    Class names are read from the README.txt in the dataset root.
    """

    dataset_dir = 'imagenet-r'

    def __init__(self, root, num_shots):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        readme = os.path.join(self.dataset_dir, 'README.txt')
        synset_to_name = _parse_synset_readme(readme)

        self.template = ['a photo of a {}.']
        all_items, _ = _load_synset_folder(self.dataset_dir, synset_to_name)
        train, val, test = _few_shot_split(all_items, num_shots)
        super().__init__(train_x=train, val=val, test=test)
