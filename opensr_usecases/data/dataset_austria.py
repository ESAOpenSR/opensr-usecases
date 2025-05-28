import numpy as np
import pandas as pd
import rasterio
import random
import torch
import os
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pathlib
from pathlib import Path


# read in panda file
class TIFDataset(Dataset):
    def __init__(
        self,
        data_table: str | Path = "",
        input_path='',
        target_path='',
        transform=A.Compose([A.Normalize(mean=0, std=1), ToTensorV2()]),
        phase="test",
        image_type="lr",
        mask_class=41,
        band_indices=None,
        bands=4,
    ):
        # own, defintely needed
        assert Path(data_table).exists()
        self.data = pd.read_csv(data_table)
        self.input_path = input_path
        self.target_path = target_path
        self.transform = transform
        self.image_type = image_type  # Either LR od HR
        self.mask_class = mask_class
        self.phase = phase
        # maybe, dont want to do 3band stuff
        self.bands = bands

        # assertion and validation
        assert self.image_type in ["hr", "sr", "sr_4band"]
        assert bands in [3, 4]

        # allows adding of individual sr-indexing of channel bands
        if band_indices is None:
            if self.image_type == 'hr':
                # assume hr is orthophoto
                self.band_indices = [1, 2, 3, 4]
                self.id_prefix = 'HR_ortho'
            elif self.image_type == 'sr':
                # Select bands: Red (B4), Green (B3), Blue (B2), and NIR (B8)
                self.band_indices = [4, 3, 2, 8]
                self.id_prefix = 'S2'
            elif self.image_type == 'sr_4band':
                # Select bands: Red (B4), Green (B3), Blue (B2), and NIR (B8)
                self.band_indices = [1, 2, 3, 4]
                self.id_prefix = 'S2'

    def validate_data(self):
        for i, col in self.data.iterrows():
            img, mask = Path(col[self.input_path]), Path(col[self.target_path])
            assert img.exists()
            assert mask.exists()
            # other validation
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # tiel indexing

        image_id = f"{self.id_prefix}_{self.data.loc[idx, 'id']:05d}.tif"
        mask_id = f"HR_mask_{self.data.loc[idx, 'id']:05d}.tif"

        img_profile = None
        # Load image
        with rasterio.open(Path(self.input_path) / image_id) as src:
            img = src.read(self.band_indices).astype(np.float32)
            img_profile = src.profile

        # Load mask
        with rasterio.open(Path(self.target_path) / mask_id) as src:
            mask = src.read(1).astype(np.uint8)  # Read first band only

        # Convert mask to binary if needed (Assumes 0/1 classes)
        mask = (mask == self.mask_class).astype(np.float32)

        if self.transform:
            transformed = self.transform(image=img.transpose(1, 2, 0), mask=mask)
            img_trafo = transformed["image"]
            mask_trafo = transformed["mask"]
        else:
            raise 'No transform selected: apply at least a normalization'

        return img_trafo, mask_trafo.unsqueeze(0)  # Add channel dimension to mask

