# 0. Imports ----------------------------------------------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pathlib
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from opensr_usecases import Validator

from multiprocessing import Process


def run_inference(config, run_id, dataloader, pred_type, model_pl, ckpt_base_path, val_obj):
    model = model_pl(config)
    weights_path = list(pathlib.Path(ckpt_base_path.replace("RUN", run_id)).glob('epoch=*.ckpt'))[0]
    print(f"{pred_type} checkpoint: {weights_path}")
    ckpt = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    val_obj.run_predictions(dataloader, model, pred_type=pred_type, load_pkl=True)


if __name__ == '__main__':
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
                return_metadata=True,
                use_subsample=True,
            )

    dataset_hr = TIFDataset(
                data_table=table_path,
                input_path='/data/USERS/shollend/combined_download/output/hr_orthofoto/',
                target_path=target_path,
                phase="test",
                image_type='hr',
                return_metadata=True,
                use_subsample=True,
            )

    dataset_sr = TIFDataset(
                data_table=table_path,
                input_path='/data/USERS/shollend/sentinel2/sr_inference/diffusion_simon/',
                target_path=target_path,
                phase="test",
                image_type='sr_4band',
                return_metadata=True,
                use_subsample=True,
            )

    # 1.2 Create DataLoaders
    dataloader_lr = DataLoader(dataset_lr, batch_size=4, shuffle=False)
    dataloader_hr = DataLoader(dataset_hr, batch_size=4, shuffle=False)
    dataloader_sr = DataLoader(dataset_sr, batch_size=4, shuffle=False)

    val_obj = Validator(output_folder="data_folder", device="cpu", force_recalc=False, debugging=False, mode='tif')
    # val_obj.create_metadata_file(pred_types=['LR', 'HR', 'SR'])
    # val_obj.save_results_examples(num_examples=1)

    print('inti dataloaders')
    # 2. Get Models -----------------------------------------------------------------------------------------------------
    from opensr_usecases.models.model_files import model_pl
    from omegaconf import OmegaConf
    config_path = pathlib.Path('/home/shollend/coding/building_segmentation/logs/Samuel_building_segmentation')
    ckpt_base_path = '/home/shollend/coding/building_segmentation/logs/Samuel_building_segmentation/RUN/'

    sr_run = '2025-06-02_18-02-33' # new diffusion
    sr_run = '2025-06-10_16-06-52'  # new new diffusion
    #lr_run = '2025-06-02_12-52-33' # bilinear
    lr_run = '2025-06-10_10-47-46'  # nn
    hr_run = '2025-06-02_12-53-41' # orthofoto

    lr_config = OmegaConf.load(config_path / lr_run / 'train_config.yaml')
    hr_config = OmegaConf.load(config_path / hr_run/ 'train_config.yaml')
    sr_config = OmegaConf.load(config_path / sr_run / 'train_config.yaml')

    lr_model = model_pl(lr_config)
    lr_weights = list(pathlib.Path(ckpt_base_path.replace("RUN", lr_run)).glob('epoch=*.ckpt'))[0]
    print(lr_weights)
    lr_ckpt = torch.load(lr_weights, map_location='cpu')
    lr_model.load_state_dict(lr_ckpt['state_dict'])
    val_obj.run_predictions(dataloader_lr, lr_model, pred_type="LR", load_pkl=True)

    hr_model = model_pl(hr_config)
    hr_weights = list(pathlib.Path(ckpt_base_path.replace("RUN", hr_run)).glob('epoch=*.ckpt'))[0]
    print(hr_weights)
    hr_ckpt = torch.load(hr_weights, map_location='cpu')
    hr_model.load_state_dict(hr_ckpt['state_dict'])

    val_obj.run_predictions(dataloader_hr, hr_model, pred_type="HR", load_pkl=True)

    sr_model = model_pl(sr_config)
    sr_weights = list(pathlib.Path(ckpt_base_path.replace("RUN", sr_run)).glob('epoch=*.ckpt'))[0]
    print(sr_weights)
    sr_ckpt = torch.load(sr_weights, map_location='cpu')
    sr_model.load_state_dict(sr_ckpt['state_dict'])

    val_obj.run_predictions(dataloader_sr, sr_model, pred_type="SR", load_pkl=True)

    # 3.2  Calculate images and save to Disk
    #val_obj.create_metadata_file(pred_types=['LR', 'HR', 'SR'])
    # 3.3 - Calcuate Metrics
    # 3.3.1 Calculate Segmentation Metrics based on predictions
    val_obj.calculate_segmentation_metrics(pred_type="LR", threshold=0.75)
    val_obj.calculate_segmentation_metrics(pred_type="HR", threshold=0.75)
    val_obj.calculate_segmentation_metrics(pred_type="SR", threshold=0.75)

    # 3.3.2 Calculate Object Detection Metrics based on predictions
    val_obj.calculate_object_detection_metrics(pred_type="LR", threshold=0.75)
    val_obj.calculate_object_detection_metrics(pred_type="HR", threshold=0.75)
    val_obj.calculate_object_detection_metrics(pred_type="SR", threshold=0.75)


    # 4. Check out Results and Metrics -------------------------------------------------------------------------------------

    # 4.2 Check Segmentation Metrics
    val_obj.print_segmentation_metrics(save_csv=True)
    val_obj.print_segmentation_improvements(save_csv=True)

    # 4.3 Check Object Detection Metrics
    val_obj.print_object_detection_metrics(save_csv=True)
    val_obj.print_object_detection_improvements(save_csv=True)


    val_obj.save_results_examples(num_examples=1)

    # 4.4 Check Threshold Curves
    val_obj.plot_threshold_curves(metric="all")

