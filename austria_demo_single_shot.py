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
    val_obj = ValidatorAustriaSingleShot(output_folder="data_folder/bilinear_newdiffusion", device="cpu", force_recalc=False, debugging=False, mode='tif')

    BASE_DATA = Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\data\inferred_buildings')
    gt = BASE_DATA / 'gt' / 'run_'
    lr_bilinear = BASE_DATA / 'bilinear' / 'run_2025-05-06_17-08-07'
    lr_nn = BASE_DATA / 'nn' / 'run_512'
    sr = BASE_DATA / 'diffusion_new' / 'run_'
    hr = BASE_DATA / 'baseline' / 'run_'

    for folder in [lr_bilinear, lr_nn, sr, hr]:
        metrics_folder = folder / 'metrics'
        metrics_folder.mkdir(exist_ok=True)

        metrics_folder = folder / 'metrics_debug'
        metrics_folder.mkdir(exist_ok=True)

    val_obj.create_metadata_file(gt_path=gt,
                                 lr_path=lr_bilinear / 'imgs',
                                 hr_path=hr / 'imgs',
                                 sr_path=sr / 'imgs')

    # Calculate single shot metrics
    val_obj.calculate_all_metrics(pred_type="LR", threshold=0.75)
    val_obj.calculate_all_metrics(pred_type="HR", threshold=0.75)
    val_obj.calculate_all_metrics(pred_type="SR", threshold=0.75)

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

