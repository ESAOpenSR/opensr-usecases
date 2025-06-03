# 0. Imports ----------------------------------------------------------------------------------------
from torch.utils.data import DataLoader
from tqdm import tqdm
from opensr_usecases import Validator


# 1. Get Data
# 1.1 Get Datasets
from opensr_usecases.data.placeholder_dataset import PlaceholderDataset
dataset_lr = PlaceholderDataset(phase="test", image_type="lr")
dataset_hr = PlaceholderDataset(phase="test", image_type="hr")
dataset_sr = PlaceholderDataset(phase="test", image_type="sr")

# 1.2 Create DataLoaders
dataloader_lr = DataLoader(dataset_lr, batch_size=8, shuffle=True)
dataloader_hr = DataLoader(dataset_hr, batch_size=8, shuffle=True)
dataloader_sr = DataLoader(dataset_sr, batch_size=8, shuffle=True)


# 2. Get Models -----------------------------------------------------------------------------------------------------
from opensr_usecases.models.placeholder_model import PlaceholderModel
lr_model = PlaceholderModel()
hr_model = PlaceholderModel()
sr_model = PlaceholderModel()


# 3. Validate -----------------------------------------------------------------------------------------------------
# 3.1 Create Validator object and set settings
val_obj = Validator(output_folder="data_folder", device="cpu", force_recalc= True, debugging=False)
global_threshold = 0.50

# 3.2  Calculate images and save to Disk
val_obj.run_predictions(dataloader_lr, lr_model, pred_type="LR", load_pkl=False)
val_obj.run_predictions(dataloader_hr, hr_model, pred_type="HR", load_pkl=False)
val_obj.run_predictions(dataloader_sr, sr_model, pred_type="SR", load_pkl=False)

# 3.3 - Calcuate Metrics
# 3.3.1 Calculate Segmentation Metrics based on predictions
val_obj.calculate_segmentation_metrics(pred_type="LR", threshold=global_threshold)
val_obj.calculate_segmentation_metrics(pred_type="HR", threshold=global_threshold)
val_obj.calculate_segmentation_metrics(pred_type="SR", threshold=global_threshold)
    
# 3.3.2 Calculate Object Detection Metrics based on predictions
val_obj.calculate_object_detection_metrics(pred_type="LR", threshold=global_threshold)
val_obj.calculate_object_detection_metrics(pred_type="HR", threshold=global_threshold)
val_obj.calculate_object_detection_metrics(pred_type="SR", threshold=global_threshold)

# 3.3.3 Calculate Object Detection Metrics by Object Sizes
val_obj.calculate_object_detection_metrics_by_size(pred_type="LR", threshold=global_threshold)
val_obj.calculate_object_detection_metrics_by_size(pred_type="HR", threshold=global_threshold)
val_obj.calculate_object_detection_metrics_by_size(pred_type="SR", threshold=global_threshold)

# 3.3.4 Calculate Percent of Objects Found by Size
val_obj.calculate_percent_objects_found_by_size(pred_type="LR", threshold=global_threshold)
val_obj.calculate_percent_objects_found_by_size(pred_type="HR", threshold=global_threshold)
val_obj.calculate_percent_objects_found_by_size(pred_type="SR", threshold=global_threshold)


# 4. Check out Results and Metrics -------------------------------------------------------------------------------------
# 4.1 Visual Inspection
val_obj.save_results_examples(num_examples=5)

# 4.2 Check Segmentation Metrics
val_obj.print_segmentation_metrics(save_csv=True)
val_obj.print_segmentation_improvements(save_csv=True)

# 4.3 Check Object Detection Metrics
val_obj.print_object_detection_metrics(save_csv=True)
val_obj.print_object_detection_improvements(save_csv=True)

# 4.4 Check Object Detection Metrics by Size
val_obj.print_object_detection_metrics_by_size(save_csv=True)
val_obj.print_object_detection_improvements_by_size(save_csv=True)

# 4.5 Check Object Detection Percent of Objects found - by Size
val_obj.print_percent_objects_found_by_size(save_csv=True)
val_obj.print_percent_objects_found_improvements_by_size(save_csv=True)

# 4.4 Check Threshold Curves
val_obj.plot_threshold_curves(metric="all")

