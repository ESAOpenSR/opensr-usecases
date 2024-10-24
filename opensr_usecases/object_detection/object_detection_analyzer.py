import numpy as np
from collections import defaultdict
import torch



class ObjectDetectionAnalyzer:
    """
    A class to analyze object detection predictions by computing various metrics including 
    object-specific scores and standard segmentation metrics.

    This class provides methods to compute average object prediction scores, 
    percentage of found objects, and other object-based statistics, as well as standard 
    segmentation metrics like precision, recall, and IoU.

    Attributes:
        compute_avg_object_prediction_score (function): Computes the average prediction score for objects.
        compute_found_objects_percentage (function): Computes the percentage of objects found above a certain threshold.
        compute_avg_object_prediction_score_by_size (function): Computes the average prediction score weighted by object size.
        standard_metrics (function): Computes standard segmentation metrics (precision, recall, IoU, etc.).

    """
    def __init__(self):
        # Import the required functions from the object_detection_utils module
        from opensr_usecases.object_detection.object_detection_utils import compute_avg_object_prediction_score
        from opensr_usecases.object_detection.object_detection_utils import compute_found_objects_percentage
        from opensr_usecases.object_detection.object_detection_utils import compute_avg_object_prediction_score_by_size
        from opensr_usecases.object_detection.object_detection_utils import standard_metrics
        # Assign the imported functions to instance variables
        self.compute_avg_object_prediction_score = compute_avg_object_prediction_score
        self.compute_found_objects_percentage = compute_found_objects_percentage
        self.compute_avg_object_prediction_score_by_size = compute_avg_object_prediction_score_by_size
        self.standard_metrics = standard_metrics
        
    def check_mask_validity(self, mask):
        """
        Ensures the validity and proper format of the input mask by converting it to a numpy array, 
        adjusting its dimensions, and clipping its values between 0 and 1.

        Args:
            mask (torch.Tensor or np.ndarray): The input mask which can either be a PyTorch tensor 
                                            or a NumPy array.

        Returns:
            np.ndarray: A valid mask as a NumPy array with proper dimensions and values clipped 
                        between 0 and 1.

        Notes:
            - Converts a PyTorch tensor to a NumPy array if needed.
            - Ensures the mask has a minimum of 3 dimensions if it's initially 2D.
            - Squeezes single-channel masks to remove unnecessary dimensions.
            - Clips the mask values between 0 and 1 to ensure valid data.
        """
        # If the mask is a PyTorch tensor, detach it from the computation graph and move it to CPU
        if type(mask) == torch.Tensor:
            mask = mask.detach().cpu()

        # If the mask is not a numpy array, convert it to one
        if not isinstance(mask, np.ndarray):
            mask = mask.numpy()

        # If the mask is 2D, add a dimension to make it 3D (batch dimension)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        # If the second dimension is 1 (single-channel mask), remove the unnecessary channel dimension
        if mask.shape[1] == 1:
            mask = mask.squeeze()

        # Clip the mask values between 0 and 1 to ensure valid data
        mask = mask.clip(0, 1)

        return mask
        
    def compute(self, target, pred):
        """
        Computes various segmentation metrics between the target and predicted masks, including object-specific 
        metrics and standard segmentation metrics.

        Args:
            target (torch.Tensor or np.ndarray): The ground truth mask.
            pred (torch.Tensor or np.ndarray): The predicted mask.

        Returns:
            dict: A dictionary containing various computed metrics, including:
                - "avg_obj_score": Average prediction score for all objects.
                - "perc_found_obj": Percentage of objects found above a certain threshold.
                - "avg_obj_pred_score_by_size": Average prediction score weighted by object size.
                - Standard metrics such as precision, recall, IoU, and more.

        Notes:
            - The function ensures the masks are in valid format before computing metrics.
            - It calculates object-specific metrics like average prediction score and percentage of found objects.
            - Standard segmentation metrics are computed and added to the dictionary.
        """
        # Prepare and validate the input masks
        target, pred = self.check_mask_validity(target), self.check_mask_validity(pred)
        
        # Calculate object-specific metrics and store in the dictionary
        metrics_dict = {
            "avg_obj_score": self.compute_avg_object_prediction_score(target, pred),
            "perc_found_obj": self.compute_found_objects_percentage(target, pred),
            "avg_obj_pred_score_by_size": self.compute_avg_object_prediction_score_by_size(target, pred)
        }
        
        # Calculate and add standard metrics (e.g., precision, recall, IoU) to the dictionary
        metrics_dict.update(self.standard_metrics(target, pred))
        
        # Return the complete dictionary of metrics
        return metrics_dict
    

if __name__ == "__main__":
    analyzer = ObjectDetectionAnalyzer()

    # Get some data
    from data.dataset_usa_buildings import SegmentationDataset
    ds = SegmentationDataset(phase="val",image_type="sr")
    im,mask = ds.__getitem__(10)

    # test
    metrics = analyzer.compute(target=mask, pred=np.random.rand(*mask.shape))





        