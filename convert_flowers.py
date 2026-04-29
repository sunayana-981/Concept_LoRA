# convert_flowers.py
import pandas as pd
import os
from PIL import Image
import io
from pathlib import Path

src = '/home/sunayana/datasets/flowers102/data'
dst = '/home/sunayana/datasets/flowers102_imagefolder'

for split_file in Path(src).glob('*.parquet'):
    split = 'train' if 'train' in split_file.name else 'test'
    df = pd.read_parquet(split_file)
    
    for idx, row in df.iterrows():
        label = str(row['label'])
        img_bytes = row['image']['bytes']
        
        out_dir = os.path.join(dst, split, label)
        os.makedirs(out_dir, exist_ok=True)
        
        img = Image.open(io.BytesIO(img_bytes))
        img.save(os.path.join(out_dir, f'{idx}.jpg'))
        
        if idx % 100 == 0:
            print(f"{split}: {idx} done")

print("Done!")
