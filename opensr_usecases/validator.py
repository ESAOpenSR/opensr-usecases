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
        pass

    def save_results_examples(self, num_examples=5):
        """
        Saves a few example images with their predicted masks and ground truth masks (LR, SR, HR) to disk.
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


    def calculate_segmentation_metrics(self, pred_type, threshold=0.75):
        """
        Calculates segmentation metrics for the predicted masks of a specific type (LR, HR, or SR).

        Args:
            pred_type (str): The type of prediction to calculate metrics for ("LR", "HR", or "SR").

        Returns:
            dict: A dictionary containing the computed segmentation metrics.
        """
        
        from opensr_usecases.segmentation.segmentation_utils import segmentation_metrics
        from opensr_usecases.utils.dict_average import compute_average_metrics

        # iterate over dataframe
        metrics_list = []
        for index, row in tqdm(self.metadata.iterrows(), desc=f"Calculating segmentation metrics for {pred_type}"):
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

        # Initialize or update self.segmentation_metrics
        if not hasattr(self, "segmentation_metrics") or self.segmentation_metrics is None or len(self.segmentation_metrics) == 0:
            self.segmentation_metrics = metrics_df
        else:
            self.segmentation_metrics = pd.concat([self.segmentation_metrics, metrics_df])

    
    def print_segmentation_metrics(self,save_csv=False):
        """
        Prints the segmentation metrics in a tabular format.
        This method displays the segmentation metrics calculated for different prediction types
        (LR, HR, SR) in a structured DataFrame format.
        Args:
            save_csv (bool): If True, saves the segmentation metrics DataFrame to a CSV file.
        """
        if save_csv:
            os.makedirs(os.path.join(self.output_folder, "results"), exist_ok=True)
            self.segmentation_metrics.to_csv(os.path.join(self.output_folder, "results", "segmentation_metrics.csv"))

        from opensr_usecases.utils.pretty_print_df import print_pretty_dataframe
        print_pretty_dataframe(self.segmentation_metrics, index_name="Prediction Type", float_round=6)


    def print_segmentation_improvements(self, save_csv=False):
        """
        Prints the improvements in segmentation metrics between LR, SR, and HR predictions.
        This method calculates the differences in metrics between LR, SR, and HR predictions
        and displays them in a tabular format.
        Args:
            save_csv (bool): If True, saves the comparison DataFrame to a CSV file.
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


