import io
import os
import random
from collections import defaultdict

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

template = ['a photo of a {}.']

# 101 Food-101 class names in label-index order (from HuggingFace food101 dataset)
FOOD101_CLASSES = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
    'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
    'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
    'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots',
    'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries',
    'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
    'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad',
    'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger',
    'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream',
    'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese',
    'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings',
    'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
    'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich',
    'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi',
    'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese',
    'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi',
    'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles',
]


class _ParquetSplit(Dataset):
    """Torch Dataset backed by a list of (image_bytes, label) tuples."""

    def __init__(self, rows, transform=None):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        img_bytes, label = self.rows[idx]
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class Food101:
    """Food-101: 101 food-category dataset loaded from HuggingFace Parquet files.

    Expected folder structure:
        <root>/food101/data/train-*.parquet
        <root>/food101/data/validation-*.parquet
    """

    dataset_dir = 'food101'

    def __init__(self, root, num_shots):
        data_dir = os.path.join(root, self.dataset_dir, 'data')

        train_files = sorted(
            f for f in os.listdir(data_dir)
            if f.startswith('train-') and f.endswith('.parquet')
        )
        val_files = sorted(
            f for f in os.listdir(data_dir)
            if f.startswith('validation-') and f.endswith('.parquet')
        )

        df_train = pd.concat(
            [pd.read_parquet(os.path.join(data_dir, f)) for f in train_files],
            ignore_index=True,
        )
        df_val = pd.concat(
            [pd.read_parquet(os.path.join(data_dir, f)) for f in val_files],
            ignore_index=True,
        )

        # N-shot sampling from train parquet
        by_class = defaultdict(list)
        for row in df_train.itertuples(index=False):
            by_class[row.label].append(row.image['bytes'])

        train_rows, meta_val_rows = [], []
        for label in sorted(by_class):
            imgs = by_class[label]
            random.shuffle(imgs)
            n_tr = min(num_shots, len(imgs))
            n_va = min(4, len(imgs) - n_tr)
            train_rows.extend((b, label) for b in imgs[:n_tr])
            meta_val_rows.extend((b, label) for b in imgs[n_tr:n_tr + n_va])

        test_rows = [(row.image['bytes'], row.label) for row in df_val.itertuples(index=False)]

        self.train_x = _ParquetSplit(train_rows)
        self.val = _ParquetSplit(meta_val_rows)
        self.test = _ParquetSplit(test_rows)

        self.classnames = [c.replace('_', ' ') for c in FOOD101_CLASSES]
        self.template = template
