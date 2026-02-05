import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset

try:
    import medmnist
    from medmnist import INFO
except ImportError:
    print("medmnist not installed. Install with: pip install medmnist")


class MedMNISTSplit(Dataset):
    """Wrapper to flatten labels from MedMNIST datasets and apply transforms."""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Flatten label from [n] to scalar
        if hasattr(label, '__len__'):
            label = label.item() if hasattr(label, 'item') else label[0]
        
        # Apply transform if provided
        if self.transform is not None:
            image = self.transform(image)
        
        return image, int(label)


class MedMNIST:
    def __init__(self, root_path, shots):
        self.root_path = root_path
        self.shots = shots
        
        # Load PathMNIST as example (you can change to other variants)
        self.data_flag = 'pathmnist'
        info = INFO[self.data_flag]
        
        self.DataClass = getattr(medmnist, info['python_class'])
        
        # Define transforms for CLIP
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        
        eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        
        # Wrap datasets to fix label format and apply transforms
        self.train_x = MedMNISTSplit(self.DataClass(split='train', download=True, root=root_path), transform=train_transform)
        self.val = MedMNISTSplit(self.DataClass(split='val', download=True, root=root_path), transform=eval_transform)
        self.test = MedMNISTSplit(self.DataClass(split='test', download=True, root=root_path), transform=eval_transform)
        
        # Set classnames based on the dataset
        self.classnames = [info['label'][str(i)] for i in range(len(info['label']))]
        
        # Template for text prompts
        self.template = [
            'a photo of a {}.',
            'a medical image showing {}.',
            'pathology image of {}.',
        ]

    def __len__(self):
        return len(self.train_x)
    
    def __getitem__(self, idx):
        return self.train_x[idx]