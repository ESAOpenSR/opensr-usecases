# global
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm


class Validator:
    """
    A class designed to validate object detection models by predicting masks and calculating metrics.

    The `Validator` class utilizes an object detection analyzer to compute metrics for predicted masks
    from models such as super-resolution (SR), low-resolution (LR), and high-resolution (HR) models.
    It stores computed metrics in a structured dictionary and allows for the averaging of those metrics
    across batches.

    Attributes:
        device (str): The device on which the model and tensors should be loaded ("cpu" or "cuda").
        debugging (bool): Flag to indicate whether to stop early during debugging for efficiency.
        object_analyzer (ObjectDetectionAnalyzer): An analyzer used to compute various object detection metrics.
        metrics (dict): A dictionary to store averaged evaluation metrics for different model types (e.g., LR, HR, SR).
    """

    def __init__(self, output_folder="data_folder", device="cpu", debugging=False):
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
        self.device = device
        self.debugging = debugging
        self.output_folder = output_folder
        if self.debugging:
            print(
                "Warning: Debugging Mode is active. Only 2 Batches will be processed."
            )

        # This holds the path info and later on the metrics
        self.metadata = pd.DataFrame()


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


        if load_pkl and os.path.exists(os.path.join(self.output_folder,"metadata.pkl")):
            # Load metadata from pickle file - Fast
            self.metadata = pd.read_pickle(os.path.join(self.output_folder,"metadata.pkl"))
        else:
            # Save predictions to disk
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
                    im_np = np.transpose(im[:3,:,:], (1, 2, 0))

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
            self.metadata.to_pickle(os.path.join(self.output_folder, "metadata.pkl"))
            print(f"Metadata saved to {os.path.join(self.output_folder, 'metadata.pkl')}")


    def plot_threshold_curves(self, metric="IoU"): # TODO: Implement this in Validator class
        """
        Plot segmentation performance metric as a function of threshold for different prediction types.

        This method evaluates a specified segmentation metric (e.g., IoU, F1-score) across a range of threshold values 
        for each prediction type: LR (Low Resolution), HR (High Resolution), and SR (Super Resolution). It computes the 
        metric for each threshold and generates a line plot showing how the metric varies with the threshold.

        Args:
            metric (str): Name of the segmentation metric to evaluate. Must be a column returned by 
                        `calculate_segmentation_metrics()` (e.g., "IoU", "F1").

        Side Effects:
            - Displays a matplotlib plot of the metric vs. threshold.
            - Saves the plot as 'threshold_curves.png' in `self.output_folder`.

        Notes:
            - Assumes `self.metadata` contains columns named 'pred_path_LR', 'pred_path_HR', and 'pred_path_SR'.
            - Requires that segmentation metrics for each prediction type have been precomputed or can be calculated 
            via `self.calculate_segmentation_metrics()`.
        """
        results_dict = {}
        for pred_type in tqdm(["LR", "HR", "SR"], desc="Calculating Threshold Curves..."):

            if str("pred_path_"+pred_type) not in self.metadata.columns:
                print(f"No segmentation metrics found for {pred_type}. Please calculate them first. Attempting to continue...")

            # Iterate Over Dataset with new thresholds
            thresholds = np.arange(0.1, 1.0, 0.05)
            iou_scores = []
            thresholds_lst = []
            for threshold in thresholds:
                df = self.calculate_segmentation_metrics(pred_type, threshold=threshold,return_metrics=True,verbose=False)
                metric_value = df.loc[pred_type, metric] if metric in df.columns else None
                iou_scores.append(metric_value)
                thresholds_lst.append(threshold)

            # Store results in a dictionary
            results_dict[pred_type] = {
                "thresholds": thresholds_lst,
                "scores": iou_scores
            }
        # Plot the results
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for pred_type, data in results_dict.items():
            plt.plot(data["thresholds"], data["scores"], label=pred_type)

        plt.xlabel("Threshold")
        plt.ylabel(metric)
        plt.title(f"{metric} vs Threshold")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, "threshold_curves.png"))
        plt.close()

            
    def save_results_examples(self, num_examples=5):
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
            os.makedirs(os.path.join(self.output_folder, "results"), exist_ok=True)
            self.segmentation_metrics.to_csv(os.path.join(self.output_folder, "results", "segmentation_metrics.csv"))

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

        # Build comparison DataFrame
        comparison_df = pd.DataFrame({
            "LR → SR Δ": lr_diff,
            "SR": sr_row,
            "HR → SR Δ": hr_diff
        })

        # Transpose and Print
        print_pretty_dataframe(comparison_df, index_name="Metric", float_round=6)

        if save_csv:
            os.makedirs(os.path.join(self.output_folder, "results"), exist_ok=True)
            comparison_df.to_csv(os.path.join(self.output_folder, "results", "segmentation_improvements.csv"))


