import os
import numpy as np  
from PIL import Image
import medmnist
from medmnist import INFO

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

@DATASET_REGISTRY.register()
class MedMnist(DatasetBase):
    dataset_dir = "pathmnist" 

    def __init__(self, cfg):
        # Use the actual path where MedMNIST data is stored
        data_path = "/DATA/cs22btech11053/Concept_Lora/Concept_LoRA/data"
        
        dataset_name = 'pathmnist'
        info = INFO[dataset_name]
        label_dict = info['label']
        
        DataClass = getattr(medmnist, info['python_class'])
        
        # Load data from the downloaded location
        train_dataset = DataClass(split='train', download=False, root=data_path)
        val_dataset = DataClass(split='val', download=False, root=data_path)
        test_dataset = DataClass(split='test', download=False, root=data_path)
        
        # Create image directory in a writable location
        self.image_dir = os.path.join(data_path, "pathmnist_images")
        mkdir_if_missing(self.image_dir)
        
        print(f"Converting and saving MedMNIST images to {self.image_dir}")
        
        # Convert datasets with error handling
        try:
            train = self._convert_and_save_images(train_dataset, label_dict, "train")
            val = self._convert_and_save_images(val_dataset, label_dict, "val") 
            test = self._convert_and_save_images(test_dataset, label_dict, "test")
        except Exception as e:
            print(f"Error during dataset conversion: {e}")
            raise e

        super().__init__(train_x=train, val=val, test=test)

    def _convert_and_save_images(self, dataset, label_dict, split):
        items = []
        split_dir = os.path.join(self.image_dir, split)
        mkdir_if_missing(split_dir)
        
        # Limit samples for faster processing and testing
        max_samples = len(dataset)  # Further reduce for debugging
        print(f"Processing {max_samples}/{len(dataset)} {split} samples...")
        
        for i in range(max_samples):
            try:
                img, label = dataset[i]
                
                # Debug: Print image and label info
                if i == 0:
                    print(f"Sample image type: {type(img)}")
                    if hasattr(img, 'shape'):
                        print(f"Sample image shape: {img.shape}")
                    if hasattr(img, 'dtype'):
                        print(f"Sample image dtype: {img.dtype}")
                    print(f"Sample label: {label}, type: {type(label)}")
                
                # Convert to RGB PIL image with proper handling for both numpy arrays and PIL images
                if isinstance(img, np.ndarray):
                    # Ensure image is in proper range
                    if img.dtype != np.uint8:
                        # Normalize to 0-255 range if needed
                        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                    
                    if len(img.shape) == 2:  # Grayscale HW
                        pil_img = Image.fromarray(img, mode='L').convert('RGB') 
                    elif len(img.shape) == 3:
                        if img.shape[2] == 1:  # Grayscale HWC
                            pil_img = Image.fromarray(img.squeeze(), mode='L').convert('RGB')
                        elif img.shape[2] == 3:  # RGB HWC
                            pil_img = Image.fromarray(img, mode='RGB')
                        else:
                            # Handle other channel configurations
                            pil_img = Image.fromarray(img[:,:,0], mode='L').convert('RGB')
                    else:
                        raise ValueError(f"Unexpected image shape: {img.shape}")
                elif isinstance(img, Image.Image):
                    # Handle PIL Image objects - convert to RGB
                    pil_img = img.convert('RGB')
                else:
                    raise ValueError(f"Unexpected image type: {type(img)}")
                
                # Save image
                impath = os.path.join(split_dir, f"{i}.jpg") 
                pil_img.save(impath)
                
                # Get label - handle different label formats
                if isinstance(label, np.ndarray):
                    label_val = int(label.item()) if label.size == 1 else int(label[0])
                elif isinstance(label, (list, tuple)):
                    label_val = int(label[0])
                else:
                    label_val = int(label)
                
                # Get label name
                if str(label_val) in label_dict:
                    label_name = label_dict[str(label_val)]
                else:
                    label_name = f"class_{label_val}"
                
                item = Datum(impath=impath, label=label_val, classname=label_name)
                items.append(item)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        print(f"Successfully processed {len(items)} {split} samples")
        return items