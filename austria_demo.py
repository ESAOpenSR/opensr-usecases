# global
import torch
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
import numpy as np
import os
from tqdm import tqdm

# local
from opensr_usecases import Validator

# Get data
# Initialize the datasets - For LR,SR,HR
from opensr_usecases.data.self_austria_buildings import AustriaDataset

# test git
dataset_lr = AustriaDataset(data_path=Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\data\stratification_tables\test.csv'),
                            inferred_data_path=Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\data\inferred_buildings'),
                            pred_folder='bilinear',
                            run='2025-05-06_17-08-07',
                            phase="test",
                            image_type="lr")

dataset_hr = AustriaDataset(data_path=Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\data\stratification_tables\test.csv'),
                            inferred_data_path=Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\data\inferred_buildings'),
                            pred_folder='orthofoto',
                            run='2025-05-06_17-06-13',
                            phase="test",
                            image_type="hr")

dataset_sr = AustriaDataset(data_path=Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\data\stratification_tables\test.csv'),
                            inferred_data_path=Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\data\inferred_buildings'),
                            pred_folder='diffusion',
                            run='2025-05-07_21-59-34',
                            phase="test",
                            image_type="sr")

# Initialize dataloaders for each dataset
dataloader_lr = DataLoader(dataset_lr, batch_size=12, shuffle=False)
dataloader_hr = DataLoader(dataset_hr, batch_size=12, shuffle=False)
dataloader_sr = DataLoader(dataset_sr, batch_size=12, shuffle=False)

# Create Validator object
val_obj = Validator(debugging=False)

# calculate metrics
val_obj.calculate_masks_metrics(dataloader=dataloader_lr, model=lr_model, pred_type="LR")
val_obj.calculate_masks_metrics(dataloader=dataloader_hr, model=hr_model, pred_type="HR")
val_obj.calculate_masks_metrics(dataloader=dataloader_sr, model=sr_model, pred_type="SR")

# retrieve metrics
metrics = val_obj.return_raw_metrics()

# prettypring metrics
val_obj.print_sr_improvement()

# get Example images
val_obj.save_pred_images(output_path="results/example_images")


