import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from torchvision import datasets
import torch
import pickle
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    RandomResizedCrop,
    ToTensor,
    Normalize,
    ColorJitter,
    RandomHorizontalFlip,
)

class MSCOCODataset(Dataset):
    def __init__(self, images_dir, annotations_file, subset, transform=None):
        self.images_dir = images_dir
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        self.images = {img['id']: img for img in data['images']}
        self.annotations = data['annotations']

        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            # y = ann['category_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        self.ids = list(self.images.keys())
        self.ids = self.ids[:int(len(self.ids) * subset)]        # Do we want a random subset or is this ok?

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # y = self.img_to_anns[img_id][0]['category_id']
        return image, img_id

class CUBLoader(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = data_dir
        self.is_train = split == "train"
        self.data = []
        data_path = os.path.join(self.data_dir, f"{split}.pkl")

        if os.path.exists(data_path):
            self.data = pickle.load(open(data_path, "rb"))
        if transform is None:
            if split == "train":
                self.transform = Compose(
                    [
                        ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                        RandomResizedCrop(224),
                        RandomHorizontalFlip(),
                        ToTensor(),
                        Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
                    ]
                )
            else:
                self.transform = Compose(
                    [
                        Resize(size=256, interpolation=Image.BILINEAR),
                        CenterCrop(224),
                        ToTensor(),
                        Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2]),
                    ]
                )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data["img_path"]

        try:
            img_path = os.path.join(self.data_dir, img_path.split("/CUB_200_2011/")[-1])
            img = Image.open(img_path).convert("RGB")
        except:
            img_path_split = img_path.split("/")
            split = "train" if self.is_train else "test"
            img_path = "/".join(img_path_split[:2] + [split] + img_path_split[2:])
            img = Image.open(img_path).convert("RGB")

        class_label = img_data["class_label"]
        if self.transform:
            img = self.transform(img)

        return img, class_label
    
    
def get_dataloader(dataset, images_dir, annotations_file, subset=1, transform=None, batch_size=32, shuffle=True, num_workers=4):
    if dataset == "mscoco":
        dataset = MSCOCODataset(images_dir, annotations_file, subset, transform)
        return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    elif dataset == "cifar100":
        dataset = datasets.CIFAR100(root="/data1/ai22resch11001/projects/data/cifar100", train=True, download=True, transform=transform)
        return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    elif dataset == "cub":
        dataset = CUBLoader(data_dir=images_dir, split="train", transform=transform)
        return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


