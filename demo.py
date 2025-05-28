# 0. Imports ----------------------------------------------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pathlib

from torch.utils.data import DataLoader
from tqdm import tqdm
from opensr_usecases import Validator


# 1. Get Data
# 1.1 Get Datasets
from opensr_usecases.data.dataset_austria import TIFDataset
table_path = '/data/USERS/shollend/metadata/stratification_tables/test.csv'
target_path = '/data/USERS/shollend/combined_download/output/hr_mask/'

dataset_lr = TIFDataset(
            data_table=table_path,
            input_path='/data/USERS/shollend/sentinel2/sr_inference/bilinear/',
            target_path=target_path,
            phase="test",
            image_type='sr',
        )

dataset_hr = TIFDataset(
            data_table=table_path,
            input_path='/data/USERS/shollend/combined_download/output/hr_orthofoto/',
            target_path=target_path,
            phase="test",
            image_type='hr',
        )

dataset_sr = TIFDataset(
            data_table=table_path,
            input_path='/data/USERS/shollend/sentinel2/sr_inference/diffusion_simon/',
            target_path=target_path,
            phase="test",
            image_type='sr_4band',
        )

# 1.2 Create DataLoaders
dataloader_lr = DataLoader(dataset_lr, batch_size=4, shuffle=True)
dataloader_hr = DataLoader(dataset_hr, batch_size=4, shuffle=True)
dataloader_sr = DataLoader(dataset_sr, batch_size=4, shuffle=True)

print('inti dataloaders')
# 2. Get Models -----------------------------------------------------------------------------------------------------
from opensr_usecases.models.model_files import model_pl
from omegaconf import OmegaConf
config_path = pathlib.Path('/home/shollend/coding/building_segmentation/logs/Samuel_building_segmentation')
lr_config = OmegaConf.load(config_path / '2025-05-06_17-08-07' / 'train_config.yaml')
hr_config = OmegaConf.load(config_path / '2025-05-06_17-06-13' / 'train_config.yaml')
sr_config = OmegaConf.load(config_path / '2025-05-07_21-59-34' / 'train_config.yaml')

lr_model = model_pl(lr_config)
hr_model = model_pl(hr_config)
sr_model = model_pl(sr_config)


# 3. Validate -----------------------------------------------------------------------------------------------------
# 3.1 Create Validator object
# cpu for friendly inference
val_obj = Validator(output_folder="data_folder", device="cuda", force_recalc= False, debugging=False)

# 3.2  Calculate images and save to Disk
val_obj.run_predictions(dataloader_lr, lr_model, pred_type="LR", load_pkl=True)
val_obj.run_predictions(dataloader_hr, hr_model, pred_type="HR", load_pkl=True)
val_obj.run_predictions(dataloader_sr, sr_model, pred_type="SR", load_pkl=True)

# 3.3 - Calcuate Metrics
# 3.3.1 Calculate Segmentation Metrics based on predictions
val_obj.calculate_segmentation_metrics(pred_type="LR", threshold=0.75)
val_obj.calculate_segmentation_metrics(pred_type="HR", threshold=0.75)
val_obj.calculate_segmentation_metrics(pred_type="SR", threshold=0.75)
    
# 3.3.2 Calculate Object Detection Metrics based on predictions
val_obj.calculate_object_detection_metrics(pred_type="LR", threshold=0.50)
val_obj.calculate_object_detection_metrics(pred_type="HR", threshold=0.50)
val_obj.calculate_object_detection_metrics(pred_type="SR", threshold=0.50)


# 4. Check out Results and Metrics -------------------------------------------------------------------------------------
# 4.1 Visual Inspection
val_obj.save_results_examples(num_examples=1)

# 4.2 Check Segmentation Metrics
val_obj.print_segmentation_metrics(save_csv=True)
val_obj.print_segmentation_improvements(save_csv=True)

# 4.3 Check Object Detection Metrics
val_obj.print_object_detection_metrics(save_csv=True)
val_obj.print_object_detection_improvements(save_csv=True)

# 4.4 Check Threshold Curves
val_obj.plot_threshold_curves(metric="all")

