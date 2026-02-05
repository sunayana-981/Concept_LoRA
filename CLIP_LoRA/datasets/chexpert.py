import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ChexpertDataset:
    def __init__(self, root_path, shots):
        self.root_path = root_path
        self.shots = shots
        
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
        
        # Load train/val/test splits
        self.train_x = ChexpertSplit(self._load_split('train'), root_path, transform=train_transform)
        self.val = ChexpertSplit(self._load_split('val'), root_path, transform=eval_transform)
        self.test = ChexpertSplit(self._load_split('test'), root_path, transform=eval_transform)
        
        # CheXpert class names (14 observations)
        self.classnames = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
        
        # Template for text prompts
        self.template = [
            'a chest x-ray showing {}.',
            'a medical image of {}.',
            'chest radiograph with {}.',
        ]
    
    def _load_split(self, split):
        csv_path = os.path.join(self.root_path, f'{split}_preprocess.csv')
        df = pd.read_csv(csv_path)
        return df.head(len(df) // self.shots * self.shots) if self.shots > 0 else df


class ChexpertSplit(Dataset):
    def __init__(self, df, root_path, transform=None):
        self.df = df
        self.root_path = root_path
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_path, row['Path'])
        image = Image.open(img_path).convert('RGB')
        label = int(row['label'])
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label