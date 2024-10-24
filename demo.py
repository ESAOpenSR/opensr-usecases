# global
import torch
from torch.utils.data import DataLoader
from PIL import Image
from numpy import np
import os
from tqdm import tqdm

# local
import opensr_usecases
from opensr_usecases import Validator

    


# Get data
# Initialize the datasets - For LR,SR,HR
from opensr_usecases.data.placeholder_dataset import PlaceholderDataset
dataset_lr = PlaceholderDataset(phase="test", image_type="lr")
dataset_hr = PlaceholderDataset(phase="test", image_type="hr")
dataset_sr = PlaceholderDataset(phase="test", image_type="sr")

# Initialize dataloaders for each dataset
dataloader_lr = DataLoader(dataset_lr, batch_size=12, shuffle=False)
dataloader_hr = DataLoader(dataset_hr, batch_size=12, shuffle=False)
dataloader_sr = DataLoader(dataset_sr, batch_size=12, shuffle=False)


# Get model
from opensr_usecases.models.placeholder_model import PlaceholderModel
lr_model = PlaceholderModel()
hr_model = PlaceholderModel()
sr_model = PlaceholderModel()

# Create Validator object
val_obj = Validator()

# calculate metrics
val_obj.calculate_masks_metrics(dataloader=dataloader_lr, model=lr_model, pred_type="LR", debugging=True)
val_obj.calculate_masks_metrics(dataloader=dataloader_hr, model=hr_model, pred_type="HR", debugging=True)
val_obj.calculate_masks_metrics(dataloader=dataloader_sr, model=sr_model, pred_type="SR", debugging=True)

# retrieve metrics
metrics = val_obj.return_raw_metrics()

# prettypring metrics
val_obj.print_sr_improvement()


            
                                
            
    