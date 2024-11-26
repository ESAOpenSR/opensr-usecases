# global
import torch

from PIL import Image
import numpy as np

# from tqdm import tqdm
import matplotlib.pyplot as plt

import io

from tqdm import tqdm

# local
from opensr_usecases.utils.utils import compute_average_metrics


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

    def __init__(self, device="cpu", debugging=False):
        """
        Initializes the `Validator` class by setting the device, debugging flag, loading the object
        detection analyzer, and preparing a metrics dictionary to store evaluation results.

        Args:
            device (str, optional): The device to use for computation ("cpu" or "cuda"). Defaults to "cpu".
            debugging (bool, optional): If set to True, will limit iterations for debugging purposes. Defaults to False.

        Attributes:
            device (str): Device to be used for model evaluation (e.g., "cuda" or "cpu").
            debugging (bool): Flag indicating if debugging mode is active.
            object_analyzer (ObjectDetectionAnalyzer): Initializes the object detection analyzer for use in metrics computation.
            metrics (dict): Initializes an empty dictionary to hold evaluation metrics for different prediction types.
        """
        self.device = device
        self.debugging = debugging
        if self.debugging:
            print(
                "Warning: Debugging Mode is active. Only 10 batches will be processed."
            )

        # Load the object detection analyzer
        from .object_detection.object_detection_analyzer import ObjectDetectionAnalyzer

        self.object_analyzer = ObjectDetectionAnalyzer()

        # Initialize an empty dictionary to store metrics for various prediction types (LR, HR, SR)
        self.metrics = {}
        self.mAP_metrics = {}

    def print_raw_metrics(self):
        """
        Prints the raw metrics stored in the object.
        """
        if len(self.metrics.keys()) == 0:
            print("No metrics have been computed yet.")
        for k in self.metrics.keys():
            print(k, "\n", self.metrics[k], "\n")

    def print_sr_improvement(self, save_to_txt=False):
        from .utils.pretty_print_metrics import print_sr_improvement

        self.print_sr_improvement = print_sr_improvement(self.metrics,save_to_txt)

    def return_raw_metrics(self):
        """
        Returns the raw metrics stored in the object.
        """
        if len(self.metrics.keys()) == 0:
            print("No metrics have been computed yet. Returning 'None'.")
            return None
        else:
            return self.metrics

    def calculate_masks_metrics(self, dataloader, model, pred_type, threshold=0.75):
        """
        Predicts masks for a given dataset using the provided model and computes relevant metrics.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader that provides batches of input images and ground truth masks.
            model (torch.nn.Module): Model used to predict the masks.
            pred_type (str): Type of prediction (e.g., "LR", "HR", "SR"). Must be one of ["LR", "HR", "SR"].

        Returns:
            dict: A dictionary containing the average metrics computed over all batches.

        Raises:
            AssertionError: If pred_type is not one of ["LR", "HR", "SR"].
        """

        # Ensure that the prediction type is valid
        assert pred_type in [
            "LR",
            "HR",
            "SR",
        ], "prediction type must be in ['LR', 'HR', 'SR']"

        # Set the model to evaluation mode and move it to the GPU (if available)
        model = model.eval().to(self.device)

        # Initialize an empty list to store metrics for each batch
        metrics_list = []

        # Disable gradient computation for faster inference
        with torch.no_grad():
            # Iterate over batches of images and ground truth masks
            image_count = 10 if self.debugging else len(dataloader)
            for id, batch in enumerate(
                tqdm(
                    dataloader,
                    desc=f"Predicting masks and calculating metrics for {pred_type}",
                    total=image_count,
                )
            ):
                # Unpack the batch (images and ground truth masks)
                images, gt_masks = batch

                # Move images to Device
                images = images.to(self.device)

                # Forward pass through the model to predict masks
                pred_masks = model(images)

                # Compute metrics using the object analyzer
                batch_metrics = self.object_analyzer.compute(
                    target=gt_masks, pred=pred_masks
                )

                # Append the computed metrics to the list
                metrics_list.append(batch_metrics)

                # Optional: Break the loop after 10 batches (for debugging or testing purposes)
                if self.debugging and id == 10:
                    break

        # Compute the average of all metrics across the batches
        averaged_metrics = compute_average_metrics(metrics_list)

        # Store the averaged metrics for the specified prediction type
        self.metrics[pred_type] = averaged_metrics

    def get_mAP_curve(self, dataloader, model, pred_type="LR", amount_batches=50):
        model = model.eval().to(self.device)

        found_percentages = []
        thresholds = np.linspace(0, 1, 50)

        percentage_batch = []

        # Run the prediction once for the entire batch
        for i in tqdm(
            range(amount_batches), desc="Computing mAP curve for " + pred_type
        ):  # Assuming 10 iterations or batches
            images, gt_masks = next(iter(dataloader))
            images = images.to(self.device)
            with torch.no_grad():
                pred_masks = model(images)  # Run the prediction once

            # Ensure masks are valid
            gt_masks, pred_masks = self.object_analyzer.check_mask_validity(
                gt_masks
            ), self.object_analyzer.check_mask_validity(pred_masks)

            # Iterate over thresholds
            threshold_percentages = []
            for thresh in thresholds:
                threshold_percentages.append(
                    self.object_analyzer.compute_found_objects_percentage(
                        gt_masks, pred_masks, thresh
                    )
                )

            # Calculate the average found percentage for each threshold
            percentage_batch.append(threshold_percentages)

        # Calculate the mean across batches
        found_percentages = np.mean(percentage_batch, axis=0)

        # Store results in the mAP metrics dictionary
        self.mAP_metrics[pred_type] = {
            "thresholds": thresholds,
            "TP_percentage": found_percentages,
        }

    def plot_mAP_curve(self):
        """
        Plots and returns the mean Average Precision (mAP) curve as a PIL image.

        This method visualizes the mAP curve for different prediction types based on
        previously computed mAP metrics stored in the `self.mAP_metrics` dictionary.
        The mAP curve is plotted by comparing the confidence thresholds with the
        percentage of objects found (True Positives) at each threshold.

        If no mAP metrics are available, it prints a message guiding the user to compute
        the metrics first.

        Returns:
            PIL.Image: The plotted mAP curve as a PIL image object. If no metrics are
                    available, returns None.

        Raises:
            None. This function will handle empty data and simply return without plotting
            if no mAP metrics are present.

        Example:
            If the mAP metrics have been computed using:
            `Validator.get_mAP_curve(self, dataloader, model, pred_type='LR')`,
            calling `plot_mAP_curve()` will return a PIL image of the mAP curve, which can
            be displayed or saved.

        """
        if len(self.mAP_metrics.keys()) == 0:
            print("No mAP metrics have been computed yet.")
            print(
                "Compute with: Validator.get_mAP_curve(self,dataloader, model,pred_type='LR')"
            )
            return None

        # Initialize the figure
        plt.figure()

        # Loop over all prediction types in mAP_metrics
        for pred_type in self.mAP_metrics.keys():
            thresholds = self.mAP_metrics[pred_type]["thresholds"]
            found_percentages = self.mAP_metrics[pred_type]["TP_percentage"]

            # Plot the mAP curve
            plt.plot(
                thresholds[1:], found_percentages[1:], label=f"{pred_type} mAP curve"
            )

        # Adding labels and title
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Percentage of Found Objects")
        plt.title("mAP Curves for Different Prediction Types")
        plt.legend()
        plt.grid(True)

        # Save the plot to a buffer in PNG format
        buf = io.BytesIO()
        plt.savefig(buf, format="PNG")
        plt.close()  # Close the plt figure to free memory
        buf.seek(0)  # Move the buffer cursor to the beginning

        # Convert the buffer to a PIL image
        image = Image.open(buf)
        image.load()  # Ensure the image is fully loaded
        buf.close()  # Close the buffer after reading it

        # Return the PIL image
        return image
