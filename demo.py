# 0. Imports ----------------------------------------------------------------------------------------
# global
import torch
from torch.utils.data import DataLoader
from PIL import Image
from numpy import np
import os
from tqdm import tqdm
# local
from opensr_usecases import Validator


# 1. Get Data
# 1.1 Get Datasets
from opensr_usecases.data.placeholder_dataset import PlaceholderDataset
dataset_lr = PlaceholderDataset(phase="test", image_type="lr")
dataset_hr = PlaceholderDataset(phase="test", image_type="hr")
dataset_sr = PlaceholderDataset(phase="test", image_type="sr")

# 1.2 Create DataLoaders
dataloader_lr = DataLoader(dataset_lr, batch_size=4, shuffle=True)
dataloader_hr = DataLoader(dataset_hr, batch_size=4, shuffle=True)
dataloader_sr = DataLoader(dataset_sr, batch_size=4, shuffle=True)


# 2. Get Models -----------------------------------------------------------------------------------------------------
from opensr_usecases.models.placeholder_model import PlaceholderModel
lr_model = PlaceholderModel()
hr_model = PlaceholderModel()
sr_model = PlaceholderModel()

# 3. Validate -----------------------------------------------------------------------------------------------------
# 3.1 Create Validator object
val_obj = Validator(debugging=True)

# 3.2  Calculate images and save to Disk
val_obj.run_predictions(dataloader_lr, lr_model, pred_type="LR", load_pkl=True)
val_obj.run_predictions(dataloader_hr, hr_model, pred_type="HR", load_pkl=True)
val_obj.run_predictions(dataloader_sr, sr_model, pred_type="SR", load_pkl=True)

# 3.3 Calculate Segmentation Metrics based on predictions
val_obj.calculate_segmentation_metrics(pred_type="LR", threshold=0.75)
val_obj.calculate_segmentation_metrics(pred_type="HR", threshold=0.75)
val_obj.calculate_segmentation_metrics(pred_type="SR", threshold=0.75)
# 3.3.1 Print Segmentation Metrics to console
val_obj.print_segmentation_metrics(save_csv=True)
val_obj.print_segmentation_improvements(save_csv=True)

# 3.4 Calculate Object Detection Metrics based on predictions
val_obj.calculate_object_detection_metrics(pred_type="LR", threshold=0.75)
val_obj.calculate_object_detection_metrics(pred_type="HR", threshold=0.75)
val_obj.calculate_object_detection_metrics(pred_type="SR", threshold=0.75)
# 3.4.1 Print Object Detection Metrics to console
val_obj.print_object_detection_metrics()


