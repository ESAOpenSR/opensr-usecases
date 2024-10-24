# global
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

# local
from .utils.utils import compute_average_metrics




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

        # Load the object detection analyzer
        from .object_detection.object_detection_analyzer import ObjectDetectionAnalyzer
        self.object_analyzer = ObjectDetectionAnalyzer()
        
        # Initialize an empty dictionary to store metrics for various prediction types (LR, HR, SR)
        self.metrics = {}
        
        
    def print_raw_metrics(self):
        """
        Prints the raw metrics stored in the object.
        """
        if len(self.metrics.keys()) == 0:
            print("No metrics have been computed yet.")
        for k in self.metrics.keys():
            print(k, "\n", self.metrics[k], "\n")
            
    def print_sr_improvement(self):
        from .utils.pretty_print_metrics import print_sr_improvement
        self.print_sr_improvement = print_sr_improvement(self.metrics)

    def return_raw_metrics(self):
        """
        Returns the raw metrics stored in the object.
        """
        if len(self.metrics.keys()) == 0:
            print("No metrics have been computed yet. Returning 'None'.")
            return None
        else:
            return self.metrics
                
    def calculate_masks_metrics(self, dataloader, model, pred_type,debugging=False):
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
        self.debugging = debugging
        
        # Ensure that the prediction type is valid
        assert pred_type in ["LR", "HR", "SR"], "prediction type must be in ['LR', 'HR', 'SR']"
        
        # Set the model to evaluation mode and move it to the GPU (if available)
        model = model.eval().to(self.device)
        
        # Initialize an empty list to store metrics for each batch
        metrics_list = []
        
        # Disable gradient computation for faster inference
        with torch.no_grad():
            # Iterate over batches of images and ground truth masks
            for id, batch in enumerate(tqdm(dataloader, desc="Predicting masks and calculating metrics for "+pred_type+":")):
                # Unpack the batch (images and ground truth masks)
                images, gt_masks = batch
                
                # Move images to Device
                images = images.to(self.device)
                
                # Forward pass through the model to predict masks
                pred_masks = model(images)
                
                # Compute metrics using the object analyzer
                batch_metrics = self.object_analyzer.compute(target=gt_masks, pred=pred_masks)
                
                # Append the computed metrics to the list
                metrics_list.append(batch_metrics)
                
                # Optional: Break the loop after 10 batches (for debugging or testing purposes)
                if self.debugging and id >= 10:
                    break
        
        # Compute the average of all metrics across the batches
        averaged_metrics = compute_average_metrics(metrics_list)
        
        # Store the averaged metrics for the specified prediction type
        self.metrics[pred_type] = averaged_metrics
    