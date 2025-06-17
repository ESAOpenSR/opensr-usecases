# global
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict

class Validator:
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

    def __init__(self, output_folder="data_folder", device="cpu", force_recalc=False, debugging=False):
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
        self.device = device # Device to run the model on, e.g., "cuda" or "cpu"
        self.debugging = debugging # If True, limits operations to a small number of samples/batches
        self.output_folder = output_folder # Directory where results will be saved
        self.force_recalc = force_recalc # If True, forces recalculation of predictions and metrics even if they exist
        if self.debugging:
            print(
                "Warning: Debugging Mode is active. Only 2 Batches will be processed."
            )

        # This holds the path info and later on the metrics
        self.metadata = pd.DataFrame()
        
        self.size_ranges = {     # Define size ranges for grouping objects
                                '0-4': (0, 4),
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


        metadata_path = os.path.join(self.output_folder,"internal_files", "metadata.pkl")
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
            total = 2 if self.debugging else len(dataloader)
            for id, batch in enumerate(
                tqdm(
                    dataloader,
                    desc=f"Predicting masks and calculating metrics for {pred_type}",
                    total=total)):
                # Unpack the batch (images and ground truth masks)
                images, gt_masks = batch

                # Move images to Device
                images = images.to(self.device)

                # Forward pass through the model to predict masks
                pred_masks = model(images)

                for i, (im, pred, gt) in enumerate(zip(images, pred_masks, gt_masks)):
                    global_id += 1

                    # Ensure 2D mask shape
                    pred = np.squeeze(pred.cpu().numpy())
                    gt = np.squeeze(gt.cpu().numpy())

                    # Ensure 3D-shaped input image
                    im_np = im.cpu().numpy()
                    im_np = np.transpose(im_np[:3,:,:], (1, 2, 0))

                    # Save prediction
                    pred_out_name = os.path.join(output_dir, f"pred_{global_id}.npz")
                    np.savez_compressed(pred_out_name, data=pred)

                    # Save GT - does this for each type, but doesnt matter
                    gt_out_name = os.path.join(gt_dir, f"gt_{global_id}.npz")
                    # if gt doesnt exist
                    if not os.path.exists(gt_out_name):
                        np.savez_compressed(gt_out_name, data=gt)

                    # Save input image
                    im_out_name = os.path.join(output_dir, f"image_{global_id}.npz")
                    np.savez_compressed(im_out_name, data=im_np)

                    # Append paths and IDs to lists for later use
                    image_ids.append(global_id)
                    gt_paths.append(gt_out_name)
                    image_paths.append(im_out_name)
                    pred_paths.append(pred_out_name)

                # Stop after x iterations for debugging mode
                if self.debugging and id == 2:
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


    def plot_threshold_curves(self, metric="IoU"):
        """
        Plot a segmentation performance metric across threshold values for each prediction type (LR, SR, HR).

        This method computes or loads segmentation metrics (e.g., IoU, Dice, Precision, etc.) over a range 
        of binarization thresholds for different prediction types. If previously computed data exists in 
        `self.mAP_metrics` or on disk as a .pkl file, it is reused. The selected metric is then plotted 
        against the threshold values and saved as a PNG.

        Args:
            metric (str): The segmentation metric to plot. Must be one of 
                        ["IoU", "Dice", "Precision", "Recall", "Accuracy", 
                        "Average Object Prediction Score", "Percent of Buildings Found", "all"].

        Side Effects:
            - Populates or reuses `self.mAP_metrics`, a DataFrame containing metric values for all thresholds.
            - Generates and saves a line plot (`threshold_curves_<metric>.png`) in `<output_folder>/plots`.

        Raises:
            AssertionError: If the provided metric is not supported.

        Notes:
            - Assumes predicted and ground truth masks are referenced in `self.metadata`.
        """
        supported_metrics = [
            'IoU', 'Dice', 'Precision', 'Recall', 'Accuracy',
            'Average Object Prediction Score', 'Percent of Buildings Found', 'all'
        ]
        assert metric in supported_metrics, f"Metric '{metric}' not supported. Choose from {supported_metrics}"

        if metric == "all":
            print("Plotting all supported Metrics!")
            for m in [m for m in supported_metrics if m != "all"]:
                self.plot_threshold_curves(metric=m)
            return

        # File path to cached metrics
        results_folder = os.path.join(self.output_folder, "internal_files")
        os.makedirs(results_folder, exist_ok=True)
        metrics_path = os.path.join(results_folder, "threshold_metrics.pkl")

        # Try loading from file
        if os.path.exists(metrics_path) and not self.force_recalc:
            print(f"Loading mAP_metrics from cache: {metrics_path}")
            self.mAP_metrics = pd.read_pickle(metrics_path)

        elif hasattr(self, "mAP_metrics") and isinstance(self.mAP_metrics, pd.DataFrame) and not self.mAP_metrics.empty:
            print(f"Using existing mAP_metrics from memory for metric {metric}.")

        else:
            print("Calculating mAP_metrics from scratch. This might take a moment...")
            rows = []
            for pred_type in tqdm(["LR", "HR", "SR"], desc="Calculating Threshold Curves..."):
                if f"pred_path_{pred_type}" not in self.metadata.columns:
                    print(f"No segmentation masks found for {pred_type}. Please calculate them first.")
                    continue

                thresholds = np.arange(0.1, 1.0, 0.05)
                for threshold in thresholds:
                    df1 = self.calculate_segmentation_metrics(pred_type, threshold=threshold, return_metrics=True, verbose=False)
                    df2 = self.calculate_object_detection_metrics(pred_type, threshold=threshold, return_metrics=True, verbose=False)

                    combined_row = {"Prediction Type": pred_type, "Threshold": threshold}
                    combined_row.update(df1.loc[pred_type].to_dict())
                    combined_row.update(df2.loc[pred_type].to_dict())
                    rows.append(combined_row)

            self.mAP_metrics = pd.DataFrame(rows)
            self.mAP_metrics.to_pickle(metrics_path)
            print(f"Saved mAP_metrics to: {metrics_path}")

        # PLOT
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for pred_type in ["LR", "HR", "SR"]:
            df = self.mAP_metrics[self.mAP_metrics["Prediction Type"] == pred_type]
            plt.plot(df["Threshold"], df[metric], label=pred_type)

        plt.xlabel("Threshold")
        plt.ylabel(metric)
        plt.title(f"{metric} vs Threshold")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out_folder = os.path.join(self.output_folder, "plots")
        os.makedirs(out_folder, exist_ok=True)
        plt.savefig(os.path.join(out_folder, f"threshold_curves_{metric.replace(' ', '_')}.png"))
        plt.close()



    def plot_threshold_curves_o(self, metric="IoU"):
        """
        Plot a segmentation performance metric across threshold values for each prediction type (LR, SR, HR).

        This method computes or loads segmentation metrics (e.g., IoU, Dice, Precision, etc.) over a range 
        of binarization thresholds for different prediction types. If previously computed data exists in 
        `self.mAP_metrics`, it is reused to avoid recomputation. The selected metric is then plotted against 
        the threshold values and saved as a PNG.

        Args:
            metric (str): The segmentation metric to plot. Must be one of 
                        ["IoU", "Dice", "Precision", "Recall", "Accuracy"].

        Side Effects:
            - Populates or reuses `self.mAP_metrics`, a DataFrame containing metric values for all thresholds.
            - Generates and saves a line plot (`threshold_curves.png`) in `self.output_folder`.

        Raises:
            AssertionError: If the provided metric is not in the list of supported metrics.

        Notes:
            - Assumes predicted and ground truth masks are referenced in `self.metadata`.
            - Calls `self.calculate_segmentation_metrics()` for each threshold if results are not cached.
        """
        supported_metrics = ['IoU', 'Dice', 'Precision', 'Recall', 'Accuracy', 'Average Object Prediction Score', 'Percent of Buildings Found' ,'all']
        assert metric in supported_metrics, f"Metric '{metric}' not supported. Choose from {supported_metrics}"

        if metric == "all":
            # recursive call over all supported metrics
            print(f"Plotting all supported Metrics!")
            for m in [m for m in supported_metrics if m != "all"]: # exclude "all"
                self.plot_threshold_curves(metric=m)
            return  # <- important!

        # Check if the object already has metrics stored
        if hasattr(self, "mAP_metrics") and isinstance(self.mAP_metrics, pd.DataFrame) and not self.mAP_metrics.empty:
            print("Using existing mAP_metrics from memory for metric {metric}.")
        else:
            print("Calculating mAP_metrics from scratch. This might take a moment...")
            # Iterate over LR/SR/HR data types
            rows = []
            for pred_type in tqdm(["LR", "HR", "SR"], desc="Calculating Threshold Curves..."):

                if f"pred_path_{pred_type}" not in self.metadata.columns:
                    print(f"No segmentation masks found for {pred_type}. Please calculate them first. Attempting to continue...")

                thresholds = np.arange(0.1, 1.0, 0.05)
                for threshold in thresholds:
                    df1 = self.calculate_segmentation_metrics(pred_type, threshold=threshold, return_metrics=True, verbose=False)
                    df2 = self.calculate_object_detection_metrics(pred_type, threshold=threshold, return_metrics=True, verbose=False)

                    # Combine both rows (should be Series), and store with prediction type + threshold
                    combined_row = {"Prediction Type": pred_type, "Threshold": threshold}
                    combined_row.update(df1.loc[pred_type].to_dict())
                    combined_row.update(df2.loc[pred_type].to_dict())

                    rows.append(combined_row)

            # Save all metrics across thresholds to a DataFrame
            self.mAP_metrics = pd.DataFrame(rows)


        # CONTINUE -  Plot directly from self.mAP_metrics
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for pred_type in ["LR", "HR", "SR"]:
            df = self.mAP_metrics[self.mAP_metrics["Prediction Type"] == pred_type]
            plt.plot(df["Threshold"], df[metric], label=pred_type)

        plt.xlabel("Threshold")
        plt.ylabel(metric)
        plt.title(f"{metric} vs Threshold")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out_folder = os.path.join(self.output_folder, "plots")
        os.makedirs(out_folder, exist_ok=True)
        plt.savefig(os.path.join(out_folder, str("threshold_curves_{}.png".format(metric)).replace(" ", "_") ))
        plt.close()

        
    def save_results_examples(self, num_examples=5,threshold=0.75):
        """
        Save example image triplets (input, prediction, ground truth) for LR, SR, and HR predictions.

        Randomly samples a specified number of image IDs from `self.metadata` and generates visualizations 
        for each, showing the input image, predicted mask, and ground truth mask for all three prediction 
        types: LR (Low Resolution), SR (Super Resolution), and HR (High Resolution). The resulting plots 
        are saved as PNG images in an 'examples' subdirectory.

        Args:
            num_examples (int): Number of example visualizations to generate.

        Side Effects:
            - Loads .npz files from paths listed in `self.metadata`.
            - Creates and saves composite comparison images to `self.output_folder/examples/`.

        Notes:
            - Assumes `self.metadata` contains columns: `image_path_<pred_type>`, `pred_path_<pred_type>`, and `gt_path`.
            - Applies min-max normalization to images before visualization.
            - Handles missing data gracefully and continues processing other examples.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        output_dir = os.path.join(self.output_folder, "examples")
        os.makedirs(output_dir, exist_ok=True)

        # Sample random image IDs (assumes unique image_id per row)
        sampled_rows = self.metadata.sample(num_examples, random_state=42)

        for index, row in sampled_rows.iterrows():
            fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12), dpi=150)
            pred_types = ["LR", "SR", "HR"]

            for i, pred_type in enumerate(pred_types):
                try:
                    image = np.load(row[f"image_path_{pred_type}"])["data"][:,:,:3]
                    pred_mask = np.load(row[f"pred_path_{pred_type}"])["data"]
                    gt_mask = np.load(row["gt_path"])["data"]

                    # thresold plot mask
                    pred_mask = (pred_mask >= threshold).astype(np.float32)

                    # Min-max stretch the image
                    image = (image - np.min(image)) / (np.max(image) - np.min(image))

                    # Plot: image, pred, gt
                    axes[i, 0].imshow(image)
                    axes[i, 0].set_title(f"{pred_type} Image")
                    #axes[i, 0].axis("off")

                    axes[i, 1].imshow(pred_mask, cmap="gray")
                    axes[i, 1].set_title(f"{pred_type} Prediction")
                    #axes[i, 1].axis("off")

                    axes[i, 2].imshow(gt_mask, cmap="gray")
                    axes[i, 2].set_title("Ground Truth")
                    #axes[i, 2].axis("off")

                except KeyError as e:
                    print(f"Missing data for {pred_type} in row {index}: {e}")
                    continue

            plt.tight_layout(pad=1)
            plt.savefig(os.path.join(output_dir, f"example_{index}.png"))
            plt.close(fig)

        print(f"Saved {num_examples} comparison images to '{output_dir}'.")


    def calculate_segmentation_metrics(self, pred_type, threshold=0.75,return_metrics=False,verbose=True):
        """
        Calculate average segmentation metrics for predicted masks of a specified prediction type.

        This method loads predicted and ground truth masks from disk, computes per-image segmentation metrics 
        using a given threshold, and aggregates them into an average metrics summary. Metrics can either be 
        returned as a DataFrame or stored in the object's `self.segmentation_metrics` attribute.

        Args:
            pred_type (str): Type of prediction to evaluate ("LR", "HR", or "SR").
            threshold (float, optional): Threshold to binarize predicted masks. Default is 0.75.
            return_metrics (bool, optional): If True, returns a DataFrame with average metrics instead of storing it.
            verbose (bool, optional): If True, displays a progress bar during computation.

        Returns:
            pd.DataFrame (optional): A single-row DataFrame indexed by `pred_type` with the average segmentation metrics, 
                                    if `return_metrics=True`.

        Side Effects:
            - Reads mask files from paths listed in `self.metadata`.
            - Updates `self.segmentation_metrics` by adding a new row for the specified `pred_type` if `return_metrics=False`.

        Notes:
            - Assumes masks are stored as `.npz` files under keys `"data"`.
            - Uses external utility functions `segmentation_metrics()` and `compute_average_metrics()` for computation.
        """
        from opensr_usecases.segmentation.segmentation_utils import segmentation_metrics
        from opensr_usecases.utils.dict_average import compute_average_metrics

        # iterate over dataframe
        metrics_list = []
        for index, row in tqdm(self.metadata.iterrows(), desc=f"Calculating segmentation metrics for {pred_type}", disable=not verbose):
            pred_path = row[f"pred_path_{pred_type}"]
            gt_path = row["gt_path"]

            # Load predicted and ground truth masks
            pred_mask = np.load(pred_path)["data"]
            gt_mask = np.load(gt_path)["data"]
            
            # add check to see if there are GT obeservations
            if np.sum(gt_mask) == 0:
                continue

            # Get Results Dict and append to metrics_list
            metrics = segmentation_metrics(gt_mask, pred_mask, threshold=threshold)
            metrics_list.append({k: v[0] for k, v in metrics.items()})  # flatten since we do one image per call

        # Get average over Patches
        average_metrics = compute_average_metrics(metrics_list)
        metrics_df = pd.DataFrame([average_metrics], index=[pred_type])

        if return_metrics: # if wanted, return the metrics DataFrame
            return metrics_df
        else: # Set to Object
            # Initialize or update self.segmentation_metrics
            if not hasattr(self, "segmentation_metrics") or self.segmentation_metrics is None or len(self.segmentation_metrics) == 0:
                self.segmentation_metrics = metrics_df
            else:
                self.segmentation_metrics = pd.concat([self.segmentation_metrics, metrics_df])

    
    def print_segmentation_metrics(self,save_csv=False):
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
            self.segmentation_metrics.to_csv(os.path.join(self.output_folder, "numeric_results", "segmentation_metrics.csv"))

        from opensr_usecases.utils.pretty_print_df import print_pretty_dataframe
        print_pretty_dataframe(self.segmentation_metrics, index_name="Prediction Type", float_round=6)


    def print_segmentation_improvements(self, save_csv=False):
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

        df = self.segmentation_metrics
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
        print_pretty_dataframe(comparison_df, index_name="Metric", float_round=6)

        if save_csv:
            os.makedirs(os.path.join(self.output_folder, "numeric_results"), exist_ok=True)
            comparison_df.to_csv(os.path.join(self.output_folder, "numeric_results", "segmentation_improvements.csv"))


    def calculate_object_detection_metrics(self, pred_type, threshold=0.75,return_metrics=False, verbose=False):
        """
        Calculate object detection metrics for predicted masks of a specified prediction type.

        This method computes object detection metrics such as mAP (mean Average Precision) for the predicted masks 
        of a given prediction type (e.g., LR, HR, SR). It uses the `ObjectDetectionAnalyzer` to perform the calculations.

        Args:
            pred_type (str): Type of prediction to evaluate ("LR", "HR", or "SR").
            threshold (float, optional): Threshold to binarize predicted masks. Default is 0.75.

        Returns:
            pd.DataFrame: A DataFrame containing the calculated object detection metrics.

        Raises:
            AssertionError: If `pred_type` is not one of the expected values.
        """
        # Ensure that the prediction type is valid
        assert pred_type in ["LR","HR","SR",], "prediction type must be in ['LR', 'HR', 'SR']"
        from opensr_usecases.object_detection.object_detection_utils import compute_avg_object_prediction_score
        from opensr_usecases.object_detection.object_detection_utils import compute_found_objects_percentage

        scores = []
        percentage_images_found = []
        for _, row in tqdm(self.metadata.iterrows(), desc=f"Calculating object detection metrics for {pred_type}", disable=not verbose):
            pred_path = row[f"pred_path_{pred_type}"]
            gt_path = row["gt_path"]

            pred_mask = np.load(pred_path)["data"]
            gt_mask = np.load(gt_path)["data"]
            
            # add check to see if there are GT obeservations
            if np.sum(gt_mask) == 0:
                continue

            avg_score = compute_avg_object_prediction_score(gt_mask, pred_mask)
            percentage_images_found.append(compute_found_objects_percentage(gt_mask, pred_mask, confidence_threshold=threshold))
            scores.append(avg_score)

        # Compute mean of collected scores
        avg_result = {
            "Average Object Prediction Score": np.mean(scores),
            "Percent of Buildings Found": np.mean(percentage_images_found),
        }
        df = pd.DataFrame([avg_result], index=[pred_type])

        if return_metrics:  # if wanted, return the metrics DataFrame
            return df
        else:
            # Create or update the main metrics DataFrame
            if not hasattr(self, "object_detection_metrics") or self.object_detection_metrics is None or self.object_detection_metrics.empty:
                self.object_detection_metrics = df
            else:
                self.object_detection_metrics.loc[pred_type] = df.loc[pred_type]

    
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
            self.object_detection_metrics.to_csv(os.path.join(self.output_folder, "numeric_results", "object_detection_metrics.csv"))

        from opensr_usecases.utils.pretty_print_df import print_pretty_dataframe
        print_pretty_dataframe(self.object_detection_metrics, index_name="Prediction Type", float_round=6)


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

        print_pretty_dataframe(comparison_df, index_name="Metric", float_round=6)

        if save_csv:
            os.makedirs(os.path.join(self.output_folder, "numeric_results"), exist_ok=True)
            comparison_df.to_csv(os.path.join(self.output_folder, "numeric_results", "object_detection_improvements.csv"))
            
            
                
                
    def calculate_object_detection_metrics_by_size(self, pred_type, threshold=0.75, return_metrics=False, verbose=False):
        """
        Calculate object detection metrics grouped by object size for predicted masks of a specified prediction type.

        Args:
            pred_type (str): Type of prediction to evaluate ("LR", "HR", or "SR").
            threshold (float, optional): Threshold to binarize predicted masks. Default is 0.75.
            return_metrics (bool, optional): Whether to return the resulting DataFrame.
            verbose (bool, optional): If True, shows progress bar.

        Returns:
            pd.DataFrame: DataFrame with average prediction scores per size bin (if return_metrics=True).
        """
        assert pred_type in ["LR", "HR", "SR"], "prediction type must be in ['LR', 'HR', 'SR']"
        from opensr_usecases.object_detection.object_detection_utils import compute_avg_object_prediction_score_by_size

        size_bins = self.size_ranges.keys()
        bin_scores = defaultdict(list)

        for _, row in tqdm(self.metadata.iterrows(), desc=f"Calculating size-based detection metrics for {pred_type}", disable=not verbose):
            pred_path = row[f"pred_path_{pred_type}"]
            gt_path = row["gt_path"]

            pred_mask = np.load(pred_path)["data"]
            gt_mask = np.load(gt_path)["data"]
            
            # add check to see if there are GT obeservations
            if np.sum(gt_mask) == 0:
                continue

            bin_avg_scores = compute_avg_object_prediction_score_by_size(gt_mask, pred_mask, size_ranges=self.size_ranges, threshold=threshold)

            for bin_name in size_bins:
                val = bin_avg_scores.get(bin_name)
                if val is not None:
                    bin_scores[bin_name].append(val)

        # Aggregate averages per size bin
        result = {bin_name: np.mean(bin_scores[bin_name]) if bin_scores[bin_name] else None for bin_name in size_bins}
        df = pd.DataFrame([result], index=[pred_type])

        if return_metrics:
            return df
        else:
            if not hasattr(self, "object_detection_metrics_by_size") or self.object_detection_metrics_by_size is None or self.object_detection_metrics_by_size.empty:
                self.object_detection_metrics_by_size = df
            else:
                self.object_detection_metrics_by_size.loc[pred_type] = df.loc[pred_type]
                

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
            print_pretty_dataframe(self.object_detection_metrics_by_size, index_name="Prediction Type", float_round=6)
            if save_csv:
                self.object_detection_metrics_by_size.to_csv(os.path.join(results_dir, "object_detection_metrics_by_size.csv"))


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
                comparison_df.to_csv(os.path.join(results_dir, f"object_detection_improvements_{label.lower().replace(' ', '_')}.csv"))

        # Size-based metrics
        if hasattr(self, "object_detection_metrics_by_size") and self.object_detection_metrics_by_size is not None:
            compute_and_print_deltas(self.object_detection_metrics_by_size, label="Metrics by Object Size")



    def calculate_percent_objects_found_by_size(self, pred_type, threshold=0.75, return_metrics=False, verbose=False):
        """
        Calculate percentage of objects found per size bin for a given prediction type.

        Args:
            pred_type (str): Type of prediction to evaluate ("LR", "HR", or "SR").
            threshold (float, optional): Threshold to binarize predicted masks. Default is 0.75.
            return_metrics (bool, optional): Whether to return the resulting DataFrame.
            verbose (bool, optional): If True, shows progress bar.

        Returns:
            pd.DataFrame: DataFrame with percent of objects found per size bin (if return_metrics=True).
        """
        assert pred_type in ["LR", "HR", "SR"], "prediction type must be in ['LR', 'HR', 'SR']"
        from opensr_usecases.object_detection.object_detection_utils import compute_found_objects_percentage_by_size

        size_bins = self.size_ranges.keys()
        bin_percents = defaultdict(list)

        for _, row in tqdm(self.metadata.iterrows(), desc=f"Calculating % objects found per size bin for {pred_type}", disable=not verbose):
            pred_path = row[f"pred_path_{pred_type}"]
            gt_path = row["gt_path"]

            pred_mask = np.load(pred_path)["data"]
            gt_mask = np.load(gt_path)["data"]
            
            # add check to see if there are GT obeservations
            if np.sum(gt_mask) == 0:
                continue

            bin_found_percents = compute_found_objects_percentage_by_size(gt_mask, pred_mask, size_ranges=self.size_ranges, threshold=threshold)

            for bin_name in size_bins:
                val = bin_found_percents.get(bin_name)
                if val is not None:
                    bin_percents[bin_name].append(val)

        # Average percent of objects found per bin
        result = {bin_name: np.mean(bin_percents[bin_name]) if bin_percents[bin_name] else None for bin_name in size_bins}
        df = pd.DataFrame([result], index=[pred_type])

        if return_metrics:
            return df
        else:
            if not hasattr(self, "percent_objects_found_by_size") or self.percent_objects_found_by_size is None or self.percent_objects_found_by_size.empty:
                self.percent_objects_found_by_size = df
            else:
                self.percent_objects_found_by_size.loc[pred_type] = df.loc[pred_type]
                
                
                
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
            print_pretty_dataframe(self.object_detection_metrics_by_size, index_name="Prediction Type", float_round=6)
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
            comparison_df.to_csv(os.path.join(self.output_folder, "numeric_results", "percent_objects_found_by_size_improvements.csv"))
            
            
    def update_object_identification_stats_by_size(self, pred_type, threshold=0.75, verbose=False):
        """
        Update internal statistics on true/false/total counts of object detection per size bin.

        Args:
            pred_type (str): One of 'LR', 'HR', 'SR'.
            threshold (float): Threshold to binarize predicted masks. Default is 0.75.
            verbose (bool): Whether to show progress bar.
        """
        assert pred_type in ["LR", "HR", "SR"], "prediction type must be in ['LR', 'HR', 'SR']"
        from opensr_usecases.object_detection.object_detection_utils import compute_object_detection_per_instance

        # Initialize tracking dictionary if it doesn't exist
        if not hasattr(self, "object_identification_stats"):
            self.object_identification_stats = {
                bin_name: {pt: {"true": 0, "false": 0, "total": 0} for pt in ["LR", "HR", "SR"]}
                for bin_name in self.size_ranges
            }

        for _, row in tqdm(self.metadata.iterrows(), desc=f"Updating stats for {pred_type}", disable=not verbose):
            pred_path = row[f"pred_path_{pred_type}"]
            gt_path = row["gt_path"]

            pred_mask = np.load(pred_path)["data"]
            gt_mask = np.load(gt_path)["data"]

            if np.sum(gt_mask) == 0:
                continue

            # Binarize prediction
            if pred_mask.dtype != bool:
                pred_mask = pred_mask >= threshold

            # Compute per-object detection result
            # Must return list of tuples: [(size, matched: bool), ...]
            detected_objects = compute_object_detection_per_instance(gt_mask, pred_mask)

            # Assign to size bin and update counters
            for obj_size, matched in detected_objects:
                for bin_name, (low, high) in self.size_ranges.items():
                    if low <= obj_size <= high:
                        stats = self.object_identification_stats[bin_name][pred_type]
                        stats["true"]  += int(matched)
                        stats["false"] += int(not matched)
                        stats["total"] += 1
                        break

    def print_object_identification_stats_by_size(self, save_csv=False):
        """
        Display and optionally save object identification stats by size bin and prediction type.

        Groups output so that each prediction type (LR, SR, HR) appears once, with all its size bins underneath.

        Args:
            save_csv (bool): If True, saves results to <output_folder>/numeric_results/.

        Side Effects:
            - Prints a clean grouped table.
            - Saves CSV if requested.
        """
        from opensr_usecases.utils.pretty_print_df import print_pretty_dataframe

        ordered_pred_types = ["LR", "SR", "HR"]
        ordered_size_bins = list(self.size_ranges.keys())

        rows = []
        for pred_type in ordered_pred_types:
            for bin_name in ordered_size_bins:
                stats = self.object_identification_stats[bin_name][pred_type]
                total = int(stats["total"])
                true = int(stats["true"])
                false = int(stats["false"])
                percent_found = round(100 * true / total, 1) if total > 0 else None
                rows.append({
                    "Prediction Type": pred_type,
                    "Size Bin": bin_name,
                    "Total": total,
                    "True Positives": true,
                    "False Negatives": false,
                    "% Found": percent_found
                })

        df = pd.DataFrame(rows)
        df = df.astype({
            "Total": "int",
            "True Positives": "int",
            "False Negatives": "int"
        })

        print("\nObject Identification Stats by Size Bin:")
        print_pretty_dataframe(
            df.set_index(["Prediction Type", "Size Bin"]),
            float_round=1
        )

        if save_csv:
            results_dir = os.path.join(self.output_folder, "numeric_results")
            os.makedirs(results_dir, exist_ok=True)
            df.to_csv(os.path.join(results_dir, "object_identification_stats_by_size.csv"), index=False)



    def print_object_identification_improvements_by_size(self, save_csv=False):
        """
        Display and optionally save improvement in object identification (% found) for SR over LR and HR.

        Args:
            save_csv (bool): If True, saves to <output_folder>/numeric_results/.

        Side Effects:
            - Prints comparison table.
            - Saves to disk if requested.

        Raises:
            AssertionError: If required entries are missing.
        """
        from opensr_usecases.utils.pretty_print_df import print_pretty_dataframe

        # Build % found table
        percent_found = {
            pred_type: {
                bin_name: (
                    100 * stats["true"] / stats["total"] if stats["total"] > 0 else None
                )
                for bin_name, preds in self.object_identification_stats.items()
                for pred_type_check, stats in preds.items()
                if pred_type_check == pred_type
            }
            for pred_type in ["LR", "SR", "HR"]
        }

        df = pd.DataFrame(percent_found).T  # Prediction Type as index

        assert "SR" in df.index and "LR" in df.index and "HR" in df.index, "Missing required prediction types."

        sr = df.loc["SR"]
        comparison_df = pd.DataFrame({
            "LR → SR Δ": sr - df.loc["LR"],
            "SR": sr,
            "HR → SR Δ": sr - df.loc["HR"]
        })

        print("\nImprovement in Object Identification (% Found) – SR vs LR/HR:")
        print_pretty_dataframe(comparison_df, index_name="Size Bin", float_round=2)

        if save_csv:
            results_dir = os.path.join(self.output_folder, "numeric_results")
            os.makedirs(results_dir, exist_ok=True)
            comparison_df.to_csv(os.path.join(results_dir, "object_identification_improvements_by_size.csv"))
