import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from torchvision import datasets

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
        return image

def get_dataloader(dataset, images_dir, annotations_file, subset=1, transform=None, batch_size=32, shuffle=True, num_workers=4):
    if dataset == "mscoco":
        dataset = MSCOCODataset(images_dir, annotations_file, subset, transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    if dataset == "cifar100":
        ds = datasets.CIFAR100(root="./datasets/cifar100", train=(split=="train"), download=True, transform=transform)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

