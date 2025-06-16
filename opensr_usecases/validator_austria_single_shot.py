# global
import torch
import os
import rasterio

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from affine import Affine


def load_img(src_path, dtype=np.float32, mode='tif'):
    if mode == 'npz':
        return np.load(src_path)["data"]
    else:
        with rasterio.open(src_path) as src:
            data = src.read().astype(dtype)
            # squeeze for masks
            if data.shape[0] > 1:
                return data
            else:
                return np.squeeze(data, axis=0)


def write_img(img, dest_path, dest_profile=None, mode='tif'):
    if mode == 'npz':
        np.savez_compressed(dest_path, data=img)
    else:
        if dest_profile is None:
            raise ValueError('Missing profile for tif writing')
        if img.ndim < 3:
            # add channel dimesnion
            img = np.expand_dims(img, axis=0)
        with rasterio.open(dest_path, 'w', **dest_profile) as dst:
            dst.write(img)
    return None


class ValidatorAustriaSingleShot:
    """
    Validator: A class for evaluating segmentation model outputs across multiple prediction types (LR, SR, HR).

    The `Validator` class provides a full evaluation pipeline for segmentation tasks. It handles prediction generation, 
    storage, metric calculation, and comparison across models operating at different resolutions:
    - LR: Low-Resolution input predictions
    - SR: Super-Resolution input predictions
    - HR: High-Resolution input predictions

    Key Features:
    -------------
    - Generates and saves predictions, ground truth masks, and input images as `.npz` files.
    - Automatically constructs and updates a metadata table (`self.metadata`) for all processed samples.
    - Computes per-threshold segmentation metrics (IoU, Dice, Precision, Recall, Accuracy) and plots metric-threshold curves.
    - Outputs average performance summaries per prediction type and highlights improvements of SR over LR/HR.
    - Includes optional debugging mode for rapid testing with limited batches.

    Typical Workflow:
    -----------------
    1. Use `run_predictions()` to compute and store predictions and metadata.
    2. Use `calculate_segmentation_metrics()` to compute average metrics per prediction type.
    3. Use `plot_threshold_curves()` to visualize performance variation with binarization thresholds.
    4. Use `print_segmentation_metrics()` and `print_segmentation_improvements()` to analyze results.
    5. Optionally save example comparisons using `save_results_examples()`.

    Dependencies:
    -------------
    - PyTorch
    - NumPy
    - Pandas
    - Matplotlib
    - tqdm
    - External utilities from `opensr_usecases` for metric computation and pretty-printing.

    Attributes:
    -----------
    - device (str): "cuda" or "cpu" — used to control model evaluation device.
    - debugging (bool): If True, limits operations to a small number of samples/batches.
    - output_folder (str): Directory where outputs, examples, and metadata are stored.
    - metadata (pd.DataFrame): Table containing paths to input images, predictions, and ground truths.
    - segmentation_metrics (pd.DataFrame): Stores average metrics per prediction type.
    - mAP_metrics (dict): Stores threshold curve data for supported metrics.
    """

    def __init__(self, output_folder="data_folder", device="cpu", force_recalc=False, debugging=False, mode='npz'):
        """
        Initializes the `Validator` class by setting the device, debugging flag, loading the object
        detection analyzer, and preparing a metrics dictionary to store evaluation results.

        Args:
            output_folder (str): The folder where the output predictions and metadata will be saved.
            device (str, optional): The device to use for computation ("cpu" or "cuda"). Defaults to "cpu".
            debugging (bool, optional): If set to True, will limit iterations for debugging purposes. Defaults to False.

        Attributes:
            device (str): Device to be used for model evaluation (e.g., "cuda" or "cpu").
            debugging (bool): Flag indicating if debugging mode is active.
        """
        self.device = device  # Device to run the model on, e.g., "cuda" or "cpu"
        self.debugging = debugging  # If True, limits operations to a small number of samples/batches
        self.output_folder = output_folder  # Directory where results will be saved
        self.force_recalc = force_recalc  # If True, forces recalculation of predictions and metrics even if they exist
        self.mode = mode
        if self.debugging:
            print(
                "Warning: Debugging Mode is active. Only 2 Batches will be processed."
            )

        assert mode in ['npz', 'tif', 'tiff']

        # This holds the path info and later on the metrics
        self.metadata = pd.DataFrame()

        self.metric_folder = 'metrics_debug' if self.debugging else 'metrics'

        # Define size ranges for grouping objects
        self.size_ranges = {'0-4': (0, 4),
                            '5-10': (5, 10),
                            '11-15': (11, 15),
                            '16-20': (16, 20),
                            '21-30': (21, 30),
                            '31+': (31, np.inf)}

    def run_predictions(self, dataloader, model, pred_type, load_pkl=False):
        """
        Run inference and manage prediction metadata for a specific prediction type.

        This method either loads existing prediction metadata from disk or runs the full prediction pipeline 
        (using `save_predictions`) for a given prediction type ("LR", "HR", or "SR"). It ensures that predictions 
        are generated and metadata is available for downstream evaluation or visualization.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader containing input images and ground truth masks.
            model (torch.nn.Module): Trained segmentation model to use for prediction.
            pred_type (str): One of "LR", "HR", or "SR" indicating the type of input being processed.
            load_pkl (bool, optional): If True and a metadata file exists, loads it from disk instead of re-generating predictions.

        Raises:
            AssertionError: If `pred_type` is not one of the expected values.
        """

        # Ensure that the prediction type is valid
        assert pred_type in [
            "LR",
            "HR",
            "SR",
        ], "prediction type must be in ['LR', 'HR', 'SR']"

        metadata_path = os.path.join(self.output_folder, "internal_files", "metadata.pkl")
        if load_pkl and os.path.exists(metadata_path) and not self.force_recalc:
            # Load metadata from pickle file - Fast
            print("Loading existing metadata from disk, masks have been previously calculated")
            self.metadata = pd.read_pickle(metadata_path)
        else:
            # Save predictions to disk
            print(f"Running predictions for {pred_type} and saving to disk.")
            self.save_predictions(dataloader, model, pred_type)

    def save_predictions(self, dataloader, model, pred_type):
        """
        Generate segmentation mask predictions, save results, and update metadata.

        This method performs inference using the provided model on a dataset and saves the predicted masks, ground truth masks, 
        and input images as compressed NumPy arrays. It also maintains and updates a metadata DataFrame that tracks file paths 
        for each prediction type (e.g., LR, HR, SR).

        Args:
            model (torch.nn.Module): Trained segmentation model.
            dataloader (torch.utils.data.DataLoader): DataLoader providing batches of images and ground truth masks.
            pred_type (str): Identifier for the type of prediction being processed ("LR", "HR", or "SR").

        Side Effects:
            - Creates output directories under `self.output_folder` for saving predictions, ground truths, and images.
            - Saves .npz files for predicted masks, ground truth masks (once per ID), and input images.
            - Appends paths and image IDs to internal metadata.
            - Saves metadata as a pickle file (`metadata.pkl`) when all prediction types are processed.

        Notes:
            - If `self.debugging` is True, only processes a limited number of batches.
            - Assumes `self.device` is properly set to 'cuda' or 'cpu'.
            - Assumes `self.metadata` is a pandas DataFrame with columns for different prediction types.
        """

        # 1. CHECK AND CREATE DIRECTORIES ----------------------------------------------------------------------------
        # Check if general output_res directory exists, if not create it
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # create a directory to save the predicted masks
        output_dir = os.path.join(self.output_folder, pred_type)
        os.makedirs(output_dir, exist_ok=True)

        # create GT_path
        gt_dir = os.path.join(self.output_folder, "GT")
        os.makedirs(gt_dir, exist_ok=True)

        # 1.1 Create Lists
        image_ids = []
        gt_paths = []
        image_paths = []
        pred_paths = []
        global_id = 0

        # 2. PREDICT MASKS ------------------------------------------------------------------------------------------

        # Set the model to evaluation mode and move it to the GPU (if available)
        model = model.eval().to(self.device)

        # Disable gradient computation for faster inference
        with torch.no_grad():
            # Iterate over batches of images and ground truth masks
            total = 40 if self.debugging else len(dataloader)
            for id, batch in enumerate(
                    tqdm(
                        dataloader,
                        desc=f"Predicting masks and calculating metrics for {pred_type}",
                        total=total)):
                # Unpack the batch (images and ground truth masks)
                if dataloader.dataset.return_metadata:
                    images, gt_masks, batch_image_ids = batch
                else:
                    images, gt_masks = batch
                    batch_image_ids = [global_id + i for i in range(len(images))]

                # Move images to Device
                images = images.to(self.device)

                # Forward pass through the model to predict masks
                logits = model(images)
                pred_masks = torch.sigmoid(logits)

                for i, (im, pred, gt, image_id) in enumerate(zip(images, pred_masks, gt_masks, batch_image_ids)):
                    global_id += 1

                    profile = None
                    mask_profile = None
                    # add proper image id
                    if dataloader.dataset.return_metadata:
                        image_id = f'{image_id:05d}'
                        with rasterio.open(
                                Path(dataloader.dataset.input_path) / f"{dataloader.dataset.id_prefix}_{image_id}.tif") as src_:
                            mask_profile = src_.profile

                        # adapt count and compression
                        raise TypeError(
                            'Need to impleemnt and fix the profile writing! -> switch to hr profile and fix georeferencing')
                        mask_profile.update({'count': 1, 'compress': 'zstd', 'dtype': 'float32', 'nodata': -9999})

                    # Ensure 2D mask shape
                    pred = np.squeeze(pred.cpu().numpy())
                    gt = np.squeeze(gt.cpu().numpy())

                    # Ensure 3D-shaped input image
                    im_np = im.cpu().numpy()
                    im_np = np.transpose(im_np[:3, :, :], (1, 2, 0))

                    # Save prediction
                    pred_out_name = os.path.join(output_dir, f"pred_{image_id}.{self.mode}")
                    write_img(img=pred, dest_path=pred_out_name, dest_profile=mask_profile)

                    # Save GT - does this for each type, but doesnt matter
                    gt_out_name = os.path.join(gt_dir, f"gt_{image_id}.{self.mode}")
                    # if gt doesnt exist
                    if not os.path.exists(gt_out_name):
                        write_img(img=gt, dest_path=gt_out_name, dest_profile=mask_profile)

                    # NO NEED -just link reference maybe - Save input image
                    im_out_name = os.path.join(output_dir, f"image_{image_id}.{self.mode}")
                    im_out_name = Path(dataloader.dataset.input_path) / f"{dataloader.dataset.id_prefix}_{image_id}.tif"

                    # Append paths and IDs to lists for later use
                    image_ids.append(image_id)
                    gt_paths.append(gt_out_name)
                    image_paths.append(im_out_name)
                    pred_paths.append(pred_out_name)

                # Stop after x iterations for debugging mode
                if self.debugging and id == total:
                    break

        # 3. SAVE METADATA ------------------------------------------------------------------------------------------
        df = pd.DataFrame({
            "image_id": image_ids,
            f"image_path_{pred_type}": image_paths,
            f"pred_path_{pred_type}": pred_paths,
            "gt_path": gt_paths
        })

        # Merge into self.metadata
        if self.metadata.empty:
            self.metadata = df
        else:
            self.metadata[f"image_path_{pred_type}"] = df[f"image_path_{pred_type}"]
            self.metadata[f"pred_path_{pred_type}"] = df[f"pred_path_{pred_type}"]

        # If all types have been processed, save the metadata
        if "pred_path_LR" in self.metadata.columns and "pred_path_HR" in self.metadata.columns and "pred_path_SR" in self.metadata.columns:
            out_path = os.path.join(self.output_folder, "internal_files")
            os.makedirs(out_path, exist_ok=True)
            self.metadata.to_pickle(os.path.join(out_path, "metadata.pkl"))
            print(f"Metadata saved to {os.path.join(out_path, 'metadata.pkl')}")

    def create_metadata_file(self, gt_path, lr_path, hr_path, sr_path):
        # 1st: accumulate all image_ids
        # 2nd: create lists for gt paths
        # 3rd: create lists for all ither lr/sr/hr parts

        print(f'Using: {lr_path}')
        out_path = os.path.join(self.output_folder, "internal_files")
        if os.path.exists(os.path.join(out_path, "metadata.pkl")) and os.path.exists(
                os.path.join(out_path, "metadata.csv")):
            self.metadata = pd.read_csv(os.path.join(out_path, "metadata.csv"))
            print(f'Loading metadata from: {out_path}/metadata.csv')
            return

        gt_path, lr_path, hr_path, sr_path = Path(gt_path), Path(lr_path), Path(hr_path), Path(sr_path)

        image_ids = []
        gt_paths = []
        pred_path_lr, pred_path_hr, pred_path_sr = [], [], []

        for img in tqdm(gt_path.glob(f'*.{self.mode}')):
            if img.exists():
                gt_paths.append(img)
            else:
                raise Exception('Missing: GT')

            id = img.stem.split("_")[1]
            image_ids.append(id)

            # LR
            lr_pred = lr_path / f'pred_{id}.{self.mode}'
            if lr_pred.exists():
                pred_path_lr.append(lr_pred)
            else:
                raise Exception('Missing: LR')

            # HR
            hr_pred = hr_path / f'pred_{id}.{self.mode}'
            if hr_pred.exists():
                pred_path_hr.append(hr_pred)
            else:
                raise Exception('Missing: HR')

            # LR
            sr_pred = sr_path / f'pred_{id}.{self.mode}'
            if sr_pred.exists():
                pred_path_sr.append(sr_pred)
            else:
                raise Exception('Missing: SR')

        self.metadata = pd.DataFrame({
            "image_id": image_ids,
            "gt_path": gt_paths,
            "pred_path_LR": pred_path_lr,
            "pred_path_HR": pred_path_hr,
            "pred_path_SR": pred_path_sr,
        })

        # If all types have been processed, save the metadata
        if "pred_path_LR" in self.metadata.columns and "pred_path_HR" in self.metadata.columns and "pred_path_SR" in self.metadata.columns:
            out_path = os.path.join(self.output_folder, "internal_files")
            os.makedirs(out_path, exist_ok=True)
            self.metadata.to_pickle(os.path.join(out_path, "metadata.pkl"))
            self.metadata.to_csv(os.path.join(out_path, "metadata.csv"),
                                 index=False)  # , index=True, index_label='image_id'
            print(f"Metadata saved to {os.path.join(out_path, 'metadata.pkl')}")
        return

    def save_df_wo_rewriting(self, df, save_path):
        save_path = Path(save_path)
        if save_path.exists():
            print(f'Saving dataframe: {save_path} already exists.')
        else:
            df.to_csv(save_path)
        return

    def calculate_all_metrics(self, pred_type, threshold=0.75, return_metrics=False, verbose=True):
        base = Path(self.metadata.iloc[0][f"pred_path_{pred_type}"]).parent.parent
        metric_files = [f'seg_single_img_{pred_type}.csv',
                        f'seg_global_img_{pred_type}.csv',
                        f'obj_detection_{pred_type}.csv',
                        f'obj_detection_by_size_{pred_type}.csv',
                        f'obj_found_perc_by_size_{pred_type}.csv']

        files_exist = [(base / self.metric_folder / file).exists() for file in metric_files]

        if all(files_exist):
            print(f'{pred_type}: Loading pre-computed files instead of calculating again.')
            single_image_metric = pd.read_csv(base / self.metric_folder / f'seg_single_img_{pred_type}.csv')

            metrics = single_image_metric.mean().to_dict()
            metrics_df = pd.DataFrame([metrics], index=[pred_type])
            metrics_df.index.name = 'pred_type'

            global_image_metrics = pd.read_csv(base / self.metric_folder / f'seg_global_img_{pred_type}.csv')
            global_metrics_df = self.calculate_segmentation_metrics_from_values(tp=global_image_metrics['tp'].sum(),
                                                                                fp=global_image_metrics['fp'].sum(),
                                                                                fn=global_image_metrics['fn'].sum(),
                                                                                tn=global_image_metrics['tn'].sum())
            global_metrics_df = pd.DataFrame([global_metrics_df], index=[pred_type])
            global_metrics_df.index.name = 'pred_type'

            avg_result_df = pd.read_csv(base / self.metric_folder / f'obj_detection_{pred_type}.csv', index_col='pred_type')
            obj_size_result_df = pd.read_csv(base / self.metric_folder / f'obj_detection_by_size_{pred_type}.csv', index_col='pred_type')
            obj_perc_result_df = pd.read_csv(base / self.metric_folder / f'obj_found_perc_by_size_{pred_type}.csv', index_col='pred_type')
            pass
        else:
            print(f'{pred_type}: Calculating metrics.')
            from opensr_usecases.segmentation.segmentation_utils import segmentation_metrics
            single_image_metrics = {}
            global_image_metrics = {}

            from opensr_usecases.object_detection.object_detection_utils import compute_avg_object_prediction_score
            from opensr_usecases.object_detection.object_detection_utils import compute_found_objects_percentage
            scores = []
            percentage_images_found = []

            from opensr_usecases.object_detection.object_detection_utils import \
                compute_avg_object_prediction_score_by_size
            size_bins = self.size_ranges.keys()
            bin_scores = defaultdict(list)

            from opensr_usecases.object_detection.object_detection_utils import compute_found_objects_percentage_by_size
            bin_percents = defaultdict(list)

            total = 40 if self.debugging else len(self.metadata)
            for id, (index, row) in enumerate(
                    tqdm(self.metadata.iterrows(), desc=f"Calculating ALL metrics for {pred_type}",
                         disable=not verbose, total=total)):
                pred_path = row[f"pred_path_{pred_type}"]
                gt_path = row["gt_path"]

                # Load predicted and ground truth masks
                pred_mask = load_img(pred_path)
                gt_mask = load_img(gt_path)

                # Segmentation Metrics
                metrics, metrics_global = segmentation_metrics(gt_mask, pred_mask, threshold=threshold)
                single_image_metrics[f'{int(row["image_id"]):05d}'] = {k: v[0] for k, v in metrics.items()}
                global_image_metrics[f'{int(row["image_id"]):05d}'] = {k: v[0] for k, v in metrics_global.items()}

                # Object detection
                scores.append(compute_avg_object_prediction_score(gt_mask, pred_mask))
                percentage_images_found.append(
                    compute_found_objects_percentage(gt_mask, pred_mask, confidence_threshold=threshold))

                # Object detection by size
                bin_avg_scores = compute_avg_object_prediction_score_by_size(gt_mask, pred_mask,
                                                                             size_ranges=self.size_ranges,
                                                                             threshold=threshold)

                bin_found_percents = compute_found_objects_percentage_by_size(gt_mask, pred_mask,
                                                                              size_ranges=self.size_ranges,
                                                                              threshold=threshold)

                for bin_name in size_bins:
                    val_avg = bin_avg_scores.get(bin_name)
                    if val_avg is not None:
                        bin_scores[bin_name].append(val_avg)
                    val_perc = bin_found_percents.get(bin_name)
                    if val_perc is not None:
                        bin_percents[bin_name].append(val_perc)


                if self.debugging and id == total:
                    break

            ### SEGMENTATION METRICS ###
            # save all image results
            single_image_metric = pd.DataFrame.from_dict(single_image_metrics, orient='index')
            single_image_metric.index.name = 'id'
            self.save_df_wo_rewriting(df=single_image_metric,
                                      save_path=base / self.metric_folder / f'seg_single_img_{pred_type}.csv')

            # calculate mean metrics considering the indiviudal score from each image: get averages
            metrics = single_image_metric.mean().to_dict()
            metrics_df = pd.DataFrame([metrics], index=[pred_type])
            metrics_df.index.name = 'pred_type'

            # Save all Raw image results
            global_image_metrics = pd.DataFrame.from_dict(global_image_metrics, orient='index')
            global_image_metrics.index.name = 'id'
            self.save_df_wo_rewriting(df=global_image_metrics,
                                      save_path=base / self.metric_folder / f'seg_global_img_{pred_type}.csv')

            # Calculate global segmentation metrics by Sum over all image results
            global_metrics_df = self.calculate_segmentation_metrics_from_values(tp=global_image_metrics['tp'].sum(),
                                                                                fp=global_image_metrics['fp'].sum(),
                                                                                fn=global_image_metrics['fn'].sum(),
                                                                                tn=global_image_metrics['tn'].sum())

            global_metrics_df = pd.DataFrame([global_metrics_df], index=[pred_type])
            global_metrics_df.index.name = 'pred_type'

            ### OBJECT DETECTION ###
            avg_result = {"Average Object Prediction Score": np.mean(scores),
                            "Percent of Buildings Found": np.mean(percentage_images_found),}
            avg_result_df = pd.DataFrame([avg_result], index=[pred_type])
            avg_result_df.index.name = 'pred_type'
            self.save_df_wo_rewriting(df=avg_result_df,
                                      save_path=base / self.metric_folder / f'obj_detection_{pred_type}.csv')

            ### OBJECT DETECTION BY SIZE ###
            obj_size_result = {bin_name: np.mean(bin_scores[bin_name]) if bin_scores[bin_name] else None for bin_name in
                      size_bins}
            obj_size_result_df = pd.DataFrame([obj_size_result], index=[pred_type])
            obj_size_result_df.index.name = 'pred_type'
            self.save_df_wo_rewriting(df=obj_size_result_df,
                                      save_path=base / self.metric_folder / f'obj_detection_by_size_{pred_type}.csv')

            ### OBJECT DETECTION BY SIZE PERCENTAGE ###
            obj_perc_result = {bin_name: np.mean(bin_percents[bin_name]) if bin_percents[bin_name] else None for bin_name in
                      size_bins}
            obj_perc_result_df = pd.DataFrame([obj_perc_result], index=[pred_type])
            obj_perc_result_df.index.name = 'pred_type'
            self.save_df_wo_rewriting(df=obj_perc_result_df,
                                      save_path=base / self.metric_folder / f'obj_found_perc_by_size_{pred_type}.csv')

        ### Write and save Results
        if return_metrics:
            return None
        else:
            # Single Segmentation
            if not hasattr(self, "segmentation_metrics") or self.segmentation_metrics is None or len(
                    self.segmentation_metrics) == 0:
                self.segmentation_metrics = metrics_df
            else:
                self.segmentation_metrics = pd.concat([self.segmentation_metrics, metrics_df])

            # Global Segmentation
            if not hasattr(self, "global_segmentation_metrics") or self.global_segmentation_metrics is None or len(
                    self.global_segmentation_metrics) == 0:
                self.global_segmentation_metrics = global_metrics_df
            else:
                self.global_segmentation_metrics = pd.concat([self.global_segmentation_metrics, global_metrics_df])

            # Object Detection
            if not hasattr(self,"object_detection_metrics") or self.object_detection_metrics is None or self.object_detection_metrics.empty:
                self.object_detection_metrics = avg_result_df
            else:
                self.object_detection_metrics.loc[pred_type] = avg_result_df.loc[pred_type]

            # Object Detection by Size
            if not hasattr(self, "object_detection_metrics_by_size") or self.object_detection_metrics_by_size is None or self.object_detection_metrics_by_size.empty:
                self.object_detection_metrics_by_size = obj_size_result_df
            else:
                self.object_detection_metrics_by_size.loc[pred_type] = obj_size_result_df.loc[pred_type]

            # Object Detection by Size Percentage
            if not hasattr(self,
                           "percent_objects_found_by_size") or self.percent_objects_found_by_size is None or self.percent_objects_found_by_size.empty:
                self.percent_objects_found_by_size = obj_perc_result_df
            else:
                self.percent_objects_found_by_size.loc[pred_type] = obj_perc_result_df.loc[pred_type]

        return

    def calculate_segmentation_metrics_from_values(self, tp, fp, fn, tn):
        intersection = tp
        union = tp + fp + fn
        iou = intersection / union if union > 0 else 0

        # Dice coefficient (F1 Score)
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        # Precision and Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

        return {
            "IoU": iou,
            "Dice": dice,
            'Precision': precision,
            'Recall': recall,
            'Accuracy': accuracy
        }

    def print_segmentation_metrics(self, save_csv=False):
        """
        Display and optionally save segmentation metrics for all prediction types.

        This method prints the segmentation metrics stored in `self.segmentation_metrics` in a well-formatted 
        tabular view. Optionally, the metrics can be saved to a CSV file for external use.

        Args:
            save_csv (bool): If True, saves the metrics DataFrame as a CSV file to 
                            `<output_folder>/results/segmentation_metrics.csv`.

        Side Effects:
            - Displays a table of segmentation metrics using `print_pretty_dataframe`.
            - Creates a `results` directory under `self.output_folder` if it does not exist.
            - Saves a CSV file with the metrics if `save_csv=True`.

        Notes:
            - Assumes `self.segmentation_metrics` is a populated pandas DataFrame.
            - Uses external utility `print_pretty_dataframe()` for clean formatting.
        """
        if save_csv:
            os.makedirs(os.path.join(self.output_folder, "numeric_results"), exist_ok=True)
            self.segmentation_metrics.to_csv(
                os.path.join(self.output_folder, "numeric_results", "single_segmentation_metrics.csv"))
            self.global_segmentation_metrics.to_csv(
                os.path.join(self.output_folder, "numeric_results", "global_segmentation_metrics.csv"))

        from opensr_usecases.utils.pretty_print_df import print_pretty_dataframe
        print_pretty_dataframe(self.segmentation_metrics, index_name="Prediction Type", float_round=6, table_name='Single Image Segmentation Metrics:')
        print_pretty_dataframe(self.global_segmentation_metrics, index_name="Prediction Type", float_round=6, table_name='Global Image Segmentation Metrics:')

    def print_segmentation_improvements(self, df, table_name, save_csv=False):
        """
        Display and optionally save segmentation metric improvements between LR, SR, and HR predictions.

        This method compares the segmentation performance of Super-Resolution (SR) predictions against
        Low-Resolution (LR) and High-Resolution (HR) baselines. It computes the per-metric deltas:
        - `LR → SR Δ`: Improvement from LR to SR
        - `HR → SR Δ`: Difference from HR to SR (positive means SR underperforms HR)

        The comparison is printed in a transposed tabular format and can optionally be saved as a CSV.

        Args:
            save_csv (bool): If True, saves the improvement comparison table to
                            `<output_folder>/results/segmentation_improvements.csv`.

        Side Effects:
            - Displays a formatted table showing metric differences.
            - Saves the comparison DataFrame as CSV if `save_csv=True`.

        Raises:
            AssertionError: If any of the required prediction types ("LR", "SR", "HR") are missing in `self.segmentation_metrics`.

        Notes:
            - Assumes segmentation metrics for all three prediction types have been calculated and stored.
            - Uses `print_pretty_dataframe()` for clean formatting.
        """

        from opensr_usecases.utils.pretty_print_df import print_pretty_dataframe

        assert "SR" in df.index, "SR row not found"
        assert "LR" in df.index, "LR row not found"
        assert "HR" in df.index, "HR row not found"

        sr_row = df.loc["SR"]
        lr_diff = df.loc["LR"] - sr_row
        hr_diff = df.loc["HR"] - sr_row

        # Create a DataFrame for comparison
        comparison_df = pd.DataFrame({
            "LR → SR Δ": pd.Series(lr_diff),
            "SR": pd.Series(sr_row),
            "HR → SR Δ": pd.Series(hr_diff)
        })

        # Transpose and Print
        print_pretty_dataframe(comparison_df, index_name="Metric", float_round=6, table_name=table_name)

        if save_csv:
            os.makedirs(os.path.join(self.output_folder, "numeric_results"), exist_ok=True)
            comparison_df.to_csv(os.path.join(self.output_folder, "numeric_results", "segmentation_improvements.csv"))

    def print_object_detection_metrics(self, save_csv=False):
        """
        Display and optionally save object detection metrics for all prediction types.

        This method prints the object-level detection metrics stored in `self.object_detection_metrics` in a
        well-formatted tabular view. It includes metrics such as the average object prediction score and the
        percentage of ground truth buildings correctly found based on prediction confidence.

        Args:
            save_csv (bool): If True, saves the metrics DataFrame as a CSV file to
                            `<output_folder>/results/object_detection_metrics.csv`.

        Side Effects:
            - Displays a table of object detection metrics using `print_pretty_dataframe`.
            - Creates a `results` directory under `self.output_folder` if it does not exist.
            - Saves a CSV file with the metrics if `save_csv=True`.

        Notes:
            - Assumes `self.object_detection_metrics` is a populated pandas DataFrame.
            - Uses external utility `print_pretty_dataframe()` for clean formatting.
        """
        if save_csv:
            os.makedirs(os.path.join(self.output_folder, "numeric_results"), exist_ok=True)
            self.object_detection_metrics.to_csv(
                os.path.join(self.output_folder, "numeric_results", "object_detection_metrics.csv"))

        from opensr_usecases.utils.pretty_print_df import print_pretty_dataframe
        print_pretty_dataframe(self.object_detection_metrics, index_name="Prediction Type", float_round=6, table_name='Object Detection Metric:')

    def print_object_detection_improvements(self, save_csv=False):
        """
        Display and optionally save object detection metric improvements between LR, SR, and HR predictions.

        This method compares the object detection performance of Super-Resolution (SR) predictions against
        Low-Resolution (LR) and High-Resolution (HR) baselines. It computes the per-metric deltas:
        - `LR → SR Δ`: Improvement from LR to SR
        - `HR → SR Δ`: Difference from HR to SR (positive means SR underperforms HR)

        The comparison is printed in a transposed tabular format and can optionally be saved as a CSV file.

        Args:
            save_csv (bool): If True, saves the improvement comparison table to
                            `<output_folder>/results/object_detection_improvements.csv`.

        Side Effects:
            - Displays a formatted comparison table using `print_pretty_dataframe`.
            - Saves the DataFrame as CSV if `save_csv=True`.

        Raises:
            AssertionError: If any of the required prediction types ("LR", "SR", "HR") are missing in `self.object_detection_metrics`.

        Notes:
            - Assumes `self.object_detection_metrics` contains metrics for "LR", "SR", and "HR".
            - Uses external utility `print_pretty_dataframe()` for nice formatting.
        """
        from opensr_usecases.utils.pretty_print_df import print_pretty_dataframe

        df = self.object_detection_metrics
        assert "SR" in df.index, "SR row not found"
        assert "LR" in df.index, "LR row not found"
        assert "HR" in df.index, "HR row not found"

        sr_row = df.loc["SR"]
        lr_diff = df.loc["LR"] - sr_row
        hr_diff = df.loc["HR"] - sr_row

        comparison_df = pd.DataFrame({
            "LR → SR Δ": pd.Series(lr_diff),
            "SR": pd.Series(sr_row),
            "HR → SR Δ": pd.Series(hr_diff)
        })

        print_pretty_dataframe(comparison_df, index_name="Metric", float_round=6, table_name='Object Detection Improvements:')

        if save_csv:
            os.makedirs(os.path.join(self.output_folder, "numeric_results"), exist_ok=True)
            comparison_df.to_csv(
                os.path.join(self.output_folder, "numeric_results", "object_detection_improvements.csv"))


    def print_object_detection_metrics_by_size(self, save_csv=False):
        """
        Display and optionally save segmentation and size-based object detection metrics.

        This prints the main segmentation metrics and, if available, the size-binned object detection metrics.

        Args:
            save_csv (bool): If True, saves both metrics as CSV files under <output_folder>/numeric_results/.

        Side Effects:
            - Displays tables using `print_pretty_dataframe`.
            - Saves CSVs to disk if save_csv is True.
        """
        from opensr_usecases.utils.pretty_print_df import print_pretty_dataframe
        results_dir = os.path.join(self.output_folder, "numeric_results")
        os.makedirs(results_dir, exist_ok=True)

        if hasattr(self, "object_detection_metrics_by_size") and self.object_detection_metrics_by_size is not None:
            print("\nObject Detection Metrics by Object Size:")
            print_pretty_dataframe(self.object_detection_metrics_by_size, index_name="Prediction Type",
                                   float_round=6)
            if save_csv:
                self.object_detection_metrics_by_size.to_csv(
                    os.path.join(results_dir, "object_detection_metrics_by_size.csv"))

    def print_object_detection_improvements_by_size(self, save_csv=False):
        """
        Display and optionally save object detection metric improvements between LR, SR, and HR predictions.

        Includes both global metrics and object-size-binned metrics, if available.

        Args:
            save_csv (bool): If True, saves the comparison tables as CSV files under <output_folder>/numeric_results/.

        Side Effects:
            - Displays formatted tables using `print_pretty_dataframe`.
            - Saves CSV files if requested.

        Raises:
            AssertionError: If required prediction types ("LR", "SR", "HR") are missing.
        """
        from opensr_usecases.utils.pretty_print_df import print_pretty_dataframe
        results_dir = os.path.join(self.output_folder, "numeric_results")
        os.makedirs(results_dir, exist_ok=True)

        def compute_and_print_deltas(df, label):
            assert "SR" in df.index, f"SR row not found in {label}"
            assert "LR" in df.index, f"LR row not found in {label}"
            assert "HR" in df.index, f"HR row not found in {label}"

            sr_row = df.loc["SR"]
            lr_diff = df.loc["LR"] - sr_row
            hr_diff = df.loc["HR"] - sr_row

            comparison_df = pd.DataFrame({
                "LR → SR Δ": lr_diff,
                "SR": sr_row,
                "HR → SR Δ": hr_diff,
            })

            print(f"\nObject Detection Improvements ({label}):")
            print_pretty_dataframe(comparison_df, index_name="Metric", float_round=6)

            if save_csv:
                comparison_df.to_csv(os.path.join(results_dir,
                                                  f"object_detection_improvements_{label.lower().replace(' ', '_')}.csv"))

        # Size-based metrics
        if hasattr(self, "object_detection_metrics_by_size") and self.object_detection_metrics_by_size is not None:
            compute_and_print_deltas(self.object_detection_metrics_by_size, label="Metrics by Object Size")

    def print_percent_objects_found_by_size(self, save_csv=False):
        """
        Display and optionally save size-based object detection metrics.

        This includes both:
        - Average prediction scores per object size bin.
        - Percentage of objects found per size bin.

        Args:
            save_csv (bool): If True, saves CSVs under <output_folder>/numeric_results/.

        Side Effects:
            - Displays tables using `print_pretty_dataframe`.
            - Saves CSVs to disk if save_csv is True.
        """
        from opensr_usecases.utils.pretty_print_df import print_pretty_dataframe
        results_dir = os.path.join(self.output_folder, "numeric_results")
        os.makedirs(results_dir, exist_ok=True)

        if hasattr(self, "object_detection_metrics_by_size") and self.object_detection_metrics_by_size is not None:
            print("\nAverage Prediction Score by Object Size Bin:")
            print_pretty_dataframe(self.object_detection_metrics_by_size, index_name="Prediction Type",
                                   float_round=6)
            if save_csv:
                self.object_detection_metrics_by_size.to_csv(
                    os.path.join(results_dir, "object_detection_metrics_by_size.csv")
                )

        if hasattr(self, "percent_objects_found_by_size") and self.percent_objects_found_by_size is not None:
            print("\nPercent of Objects Found by Object Size Bin:")
            print_pretty_dataframe(self.percent_objects_found_by_size, index_name="Prediction Type", float_round=2)
            if save_csv:
                self.percent_objects_found_by_size.to_csv(
                    os.path.join(results_dir, "percent_objects_found_by_size.csv")
                )

    def print_percent_objects_found_improvements_by_size(self, save_csv=False):
        """
        Display and optionally save percent-found improvements between LR, SR, and HR by object size bin.

        This compares how many objects were found per size bin, for SR vs LR and HR.

        Args:
            save_csv (bool): If True, saves the comparison table as CSV under <output_folder>/numeric_results/.

        Side Effects:
            - Prints formatted comparison table.
            - Saves CSV if requested.

        Raises:
            AssertionError: If "LR", "SR", or "HR" are missing in `self.percent_objects_found_by_size`.
        """
        from opensr_usecases.utils.pretty_print_df import print_pretty_dataframe

        df = self.percent_objects_found_by_size
        assert "SR" in df.index, "SR row not found"
        assert "LR" in df.index, "LR row not found"
        assert "HR" in df.index, "HR row not found"

        sr_row = df.loc["SR"]
        lr_diff = df.loc["LR"] - sr_row
        hr_diff = df.loc["HR"] - sr_row

        comparison_df = pd.DataFrame({
            "LR → SR Δ": lr_diff,
            "SR": sr_row,
            "HR → SR Δ": hr_diff
        })

        print("\nPercent of Objects Found by Size Bin – SR vs LR/HR:")
        print_pretty_dataframe(comparison_df, index_name="Size Bin", float_round=2)

        if save_csv:
            os.makedirs(os.path.join(self.output_folder, "numeric_results"), exist_ok=True)
            comparison_df.to_csv(os.path.join(self.output_folder, "numeric_results",
                                              "percent_objects_found_by_size_improvements.csv"))

