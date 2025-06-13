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
from opensr_usecases import Validator, ValidatorAustria
from opensr_usecases.data.dataset_austria import InferredDataset

# Initialize the datasets - For LR,SR,HR
if __name__ == '__main__':

    predict = False

    val_obj = ValidatorAustria(output_folder="data_folder/validate", device="cpu", force_recalc=False, debugging=True, mode='tif')

    if predict:
        print('inti dataloaders')
        # BILINEAR
        dataset_lr = InferredDataset(
            data_path=Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\data\stratification_tables\test.csv'),
            inferred_data_path=Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\data\inferred_buildings'),
            pred_folder='bilinear',
            run='run_2025-05-06_17-08-07',
            phase="test",
            image_type="lr")

        # NN
        # dataset_lr = InferredDataset(data_path=Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\data\stratification_tables\test.csv'),
        #                             inferred_data_path=Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\data\inferred_buildings'),
        #                             pred_folder='nn',
        #                             run='run_512',
        #                             phase="test",
        #                             image_type="lr")

        dataset_hr = InferredDataset(
            data_path=Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\data\stratification_tables\test.csv'),
            inferred_data_path=Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\data\inferred_buildings'),
            pred_folder='baseline',
            run='run_',
            phase="test",
            image_type="hr")

        dataset_sr = InferredDataset(
            data_path=Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\data\stratification_tables\test.csv'),
            inferred_data_path=Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\data\inferred_buildings'),
            pred_folder='diffusion_new',
            run='run_',
            phase="test", )

        from opensr_usecases.data.dataset_austria import TIFDataset

        # 1.2 Create DataLoaders
        dataloader_lr = DataLoader(dataset_lr, batch_size=4, shuffle=False)
        dataloader_hr = DataLoader(dataset_hr, batch_size=4, shuffle=False)
        dataloader_sr = DataLoader(dataset_sr, batch_size=4, shuffle=False)

        # 3.2  Calculate images and save to Disk
        # 2. Get Mdels -----------------------------------------------------------------------------------------------------
        from opensr_usecases.models.model_files import model_pl
        from omegaconf import OmegaConf
        config_path = pathlib.Path('/home/shollend/coding/building_segmentation/logs/Samuel_building_segmentation')
        ckpt_base_path = '/home/shollend/coding/building_segmentation/logs/Samuel_building_segmentation/RUN/'

        sr_run = '2025-06-02_18-02-33' # new diffusion
        sr_run = '2025-06-10_16-06-52'  # new new diffusion
        # lr_run = '2025-06-02_12-52-33' # bilinear wo norm
        # lr_run = '2025-05-06_17-08-07' # bilinear
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

    BASE_DATA = Path(r'C:\Users\PC\Desktop\TU\Master\MasterThesis\data\inferred_buildings')
    gt = BASE_DATA / 'gt' / 'run_'
    lr_bilinear = BASE_DATA / 'bilinear' / 'run_2025-05-06_17-08-07'
    lr_nn = BASE_DATA / 'nn' / 'run_512'
    sr = BASE_DATA / 'diffusion_new' / 'run_'
    hr = BASE_DATA / 'baseline' / 'run_'

    for folder in [lr_bilinear, lr_nn, sr, hr]:
        metrics_folder = folder / 'metrics'
        metrics_folder.mkdir(exist_ok=True)

    val_obj.create_metadata_file(gt_path=gt,
                                 lr_path=lr_bilinear / 'imgs',
                                 hr_path=hr / 'imgs',
                                 sr_path=sr / 'imgs')

    # 3.3 - Calcuate Metrics
    # 3.3.1 Calculate Segmentation Metrics based on predictions
    # individual calculationfor whole image set;
    # val_obj.calculate_segmentation_metrics(pred_type="SR", threshold=0.75)

    calc_metrics = False
    load_indivdual_metrics = True
    if load_indivdual_metrics:
        val_obj.load_individual_segmentation_metrics(lr_path=lr_bilinear,
                                                     hr_path=hr,
                                                     sr_path=sr)
    else:
        if calc_metrics:
            val_obj.calculate_segmentation_metrics(pred_type="LR", threshold=0.75)
            val_obj.calculate_segmentation_metrics(pred_type="HR", threshold=0.75)
            val_obj.calculate_segmentation_metrics(pred_type="SR", threshold=0.75)
        else:
            val_obj.segmentation_metrics = pd.read_csv(os.path.join(val_obj.output_folder, "numeric_results", "segmentation_metrics.csv"), index_col='pred_type')

    # 4.2 Check Segmentation Metrics
    val_obj.print_segmentation_metrics(save_csv=True)
    val_obj.print_segmentation_improvements(save_csv=True)

    raise TypeError

    # 3.3.2 Calculate Object Detection Metrics based on predictions
    val_obj.calculate_object_detection_metrics(pred_type="LR", threshold=0.75)
    val_obj.calculate_object_detection_metrics(pred_type="HR", threshold=0.75)
    val_obj.calculate_object_detection_metrics(pred_type="SR", threshold=0.75)


    # 4. Check out Results and Metrics -------------------------------------------------------------------------------------

    # 4.3 Check Object Detection Metrics
    val_obj.print_object_detection_metrics(save_csv=True)
    val_obj.print_object_detection_improvements(save_csv=True)


    val_obj.save_results_examples(num_examples=1)

    # 4.4 Check Threshold Curves
    val_obj.plot_threshold_curves(metric="all")



