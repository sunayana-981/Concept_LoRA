from .oxford_pets import OxfordPets
from .officehome import OfficeHome
from .kitti import KITTI
from .cityscapes import Cityscapes
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .imagenet import ImageNet
from .cub2002011 import cub2002011
from .medmnist import MedMNIST
from .chexpert import ChexpertDataset
from .chestmnist import ChestMNIST
from .imagenet_a import ImageNetA
from .imagenet_r import ImageNetR
from .imagenet_v2 import ImageNetV2
from .imagenet_sketch import ImageNetSketch


dataset_list = {
                "oxford_pets": OxfordPets,
                "eurosat": EuroSAT,
                "ucf101": UCF101,
                "sun397": SUN397,
                "caltech101": Caltech101,
                "dtd": DescribableTextures,
                "fgvc": FGVCAircraft,
                "food101": Food101,
                "oxford_flowers": OxfordFlowers,
                "stanford_cars": StanfordCars,
                "imagenet": ImageNet,
                "cub2002011": cub2002011,
                "medmnist": MedMNIST,
                "chexpert": ChexpertDataset,
                "chestmnist": ChestMNIST,
                "imagenet_a": ImageNetA,
                "imagenet_r": ImageNetR,
                "imagenet_v2": ImageNetV2,
                "imagenet_sketch": ImageNetSketch,
                "officehome": OfficeHome,
                "kitti": KITTI,
                "cityscapes": Cityscapes,
                }

# Datasets whose constructor takes (root, shots, preprocess)
_PREPROCESS_DATASETS = {'imagenet'}

# Datasets that return raw torch.utils.data.Dataset objects
# (not lists of Datum), so DataLoader is used directly
RAW_DATASETS = {'imagenet', 'chexpert', 'chestmnist', 'stanford_cars', 'food101', 'sun397',
                'cub2002011', 'officehome', 'kitti', 'cityscapes'}


def build_dataset(dataset, root_path, shots, preprocess=None):
    if dataset in _PREPROCESS_DATASETS:
        return dataset_list[dataset](root_path, shots, preprocess)
    else:
        return dataset_list[dataset](root_path, shots)
