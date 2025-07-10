# global
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
import numpy as np
import os
from tqdm import tqdm
import pathlib


# local
from opensr_usecases import Validator, ValidatorAustria, ValidatorAustriaSingleShot
from opensr_usecases.data.dataset_austria import InferredDataset

# Initialize the datasets - For LR,SR,HR
if __name__ == '__main__':
    val_obj = ValidatorAustriaSingleShot(output_folder="outputs/bilinear_diffusion_scaled_resnet18", device="cpu", force_recalc=True, debugging=False, mode='tif')

    # own
    BASE_DATA = Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\data\inferred_buildings')
    gt = BASE_DATA / 'gt' / 'run_'
    lr_bilinear = BASE_DATA / 'bilinear' / 'run_2025-05-06_17-08-07'
    lr_nn = BASE_DATA / 'nn' / 'run_512_2025-06-10_10-47-46 '
    sr_diffusion = BASE_DATA / 'diffusion_new' / 'run_2025-06-10_16-06-52'
    sr_sen2sr_lite = BASE_DATA / 'sen2sr_lite' / 'run_2025-05-12_13-18-49'
    sr_deepsent = BASE_DATA / 'deepsent' / 'run_2025-05-11_18-18-14'
    hr = BASE_DATA / 'baseline' / 'run_2025-06-02_12-53-41' ## achtung! WO NORM

    # tuwien
    BASE_DATA = Path('/home/shollend/coding/opensr-usecases')
    gt = BASE_DATA / 'GT' / 'imgs'
    hr = BASE_DATA / 'data_folder_baseline' / 'run_2025-06-02_12-53-41'

    lr_bilinear = BASE_DATA / 'data_folder_bilinear' / 'run_2025-05-06_17-08-07'
    lr_nn = BASE_DATA / 'data_folder_nn_512' / 'run_512_2025-06-10_10-47-46'
    lr_bilinear_resnet18 = BASE_DATA / 'data_folder_bilinear_scaled_resnet18' / 'run_2025-06-18_16-41-21'

    sr_diffusion = BASE_DATA / 'data_folder_diffusion_new' / 'run_2025-06-10_16-06-52'
    sr_sen2sr_lite = BASE_DATA / 'data_folder_sen2sr_lite' / 'run_2025-05-12_13-18-49'
    sr_deepsent = BASE_DATA / 'data_folder_deepsent' / 'run_2025-05-11_18-18-14'
    sr_diffusion_scaled = BASE_DATA / 'data_folder_diffusion_scaled_resnext50' / 'run_2025-06-17_15-29-13'
    sr_diffusion_scaled_resnet18 = BASE_DATA / 'data_folder_diffusion_scaled_resnet18' / 'run_2025-06-17_16-35-24'
    sr_diffusion_scaled_deeplab = BASE_DATA / 'data_folder_sr_scaled_diffusion_deeplabv3' / 'run_2025-06-18_10-47-53'

    lr = lr_bilinear_resnet18
    sr = sr_diffusion_scaled_resnet18

    for folder in [lr, sr, hr]:
        metrics_folder = folder / 'metrics'
        metrics_folder.mkdir(exist_ok=True)

        metrics_folder = folder / 'metrics_debug'
        metrics_folder.mkdir(exist_ok=True)

    val_obj.create_metadata_file(gt_path=gt,
                                 lr_path=lr / 'imgs',
                                 hr_path=hr / 'imgs',
                                 sr_path=sr / 'imgs')

    # Calculate single shot metrics
    val_obj.calculate_all_metrics(pred_type="LR", threshold=0.75)
    val_obj.calculate_all_metrics(pred_type="HR", threshold=0.75)
    val_obj.calculate_all_metrics(pred_type="SR", threshold=0.75)

    val_obj.print_all_class_metrics(save_csv=True)

    # 4.2 Check Segmentation Metrics
    val_obj.print_segmentation_metrics(save_csv=True)
    val_obj.print_segmentation_improvements(save_csv=False, df=val_obj.segmentation_metrics, table_name='Single Segmentation Metrics Improvements:')
    val_obj.print_segmentation_improvements(save_csv=False, df=val_obj.global_segmentation_metrics, table_name='Global Segmentation Metrics Improvements:')

    # 4.3 Check Object Detection Metrics
    val_obj.print_object_detection_metrics(save_csv=True)
    val_obj.print_object_detection_improvements(save_csv=True)

    # 4.4 Check Object Detection Metrics by Size
    val_obj.print_object_detection_metrics_by_size(save_csv=True)
    val_obj.print_object_detection_improvements_by_size(save_csv=True)

    # 4.5 Check Object Detection Percent of Objects found - by Size
    val_obj.print_percent_objects_found_by_size(save_csv=True)
    val_obj.print_percent_objects_found_improvements_by_size(save_csv=True)

