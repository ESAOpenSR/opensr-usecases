# global
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

# local
from opensr_usecases import Validator


# Get data
# Initialize the datasets - For LR,SR,HR
from opensr_usecases.data.placeholder_dataset import PlaceholderDataset
# test git
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
val_obj = Validator(debugging=False)

# calculate metrics
val_obj.calculate_masks_metrics(dataloader=dataloader_lr, model=lr_model, pred_type="LR")
val_obj.calculate_masks_metrics(dataloader=dataloader_hr, model=hr_model, pred_type="HR")
val_obj.calculate_masks_metrics(dataloader=dataloader_sr, model=sr_model, pred_type="SR")

# retrieve metrics
metrics = val_obj.return_raw_metrics()

# prettypring metrics
val_obj.print_sr_improvement()

# calculate mAP curves
val_obj.get_mAP_curve(dataloader_lr, lr_model, pred_type="LR", amount_batches=10)
val_obj.get_mAP_curve(dataloader_hr, hr_model, pred_type="HR", amount_batches=10)
val_obj.get_mAP_curve(dataloader_sr, sr_model, pred_type="SR", amount_batches=10)

# plot mAP curve
mAP_plot = val_obj.plot_mAP_curve()
mAP_plot.save("resources/mAP_plot.png")

# get Example images
val_obj.save_pred_images(output_path="results/example_images")

            
