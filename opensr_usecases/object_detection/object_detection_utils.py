"""
Individual Functions for Object Detection Metrics 
"""

from collections import defaultdict
import numpy as np
from scipy.ndimage import label
import torch

def compute_avg_object_prediction_score(binary_masks, predicted_masks):
    """
    Calculates the overall average prediction score for all objects across a batch of binary masks.

    Args:
        binary_masks (numpy.ndarray): A batch of binary masks of shape (batch_size, height, width), 
                                      where each distinct object is represented as a connected region 
                                      of 1s, and the background is 0.
        predicted_masks (numpy.ndarray): A batch of predicted masks of shape (batch_size, height, width), 
                                         where each pixel value represents the prediction score for that pixel.

    Returns:
        float: The overall average prediction score for all objects in the batch.
    """
    binary_masks = torch.tensor(binary_masks) if not torch.is_tensor(binary_masks) else binary_masks
    predicted_masks = torch.tensor(predicted_masks) if not torch.is_tensor(predicted_masks) else predicted_masks
    if binary_masks.ndim == 2 and predicted_masks.ndim == 2:
        predicted_masks = predicted_masks.unsqueeze(0)
        binary_masks = binary_masks.unsqueeze(0)
    binary_masks = binary_masks.cpu().numpy()
    predicted_masks = predicted_masks.cpu().numpy()
    
    total_sum = 0
    total_objects = 0
    
    batch_size = binary_masks.shape[0]
    
    for i in range(batch_size):
        binary_mask = binary_masks[i]
        predicted_mask = predicted_masks[i]
        
        labeled_mask, num_objects = label(binary_mask)
        
        # Iterate over each object in the current mask
        for object_id in range(1, num_objects + 1):
            object_mask = (labeled_mask == object_id)
            avg_value = predicted_mask[object_mask].mean()
            
            # Accumulate the sum and count of objects
            total_sum += avg_value
            total_objects += 1
    
    # Compute the overall average prediction score across all objects
    overall_avg = total_sum / total_objects if total_objects > 0 else 0
    return overall_avg


def compute_found_objects_percentage(binary_masks, predicted_masks, confidence_threshold=0.5):
    """
    Calculates the percentage of objects found based on a confidence threshold.

    Args:
        binary_masks (numpy.ndarray): A batch of binary masks of shape (batch_size, height, width), 
                                      where each distinct object is represented as a connected region 
                                      of 1s, and the background is 0.
        predicted_masks (numpy.ndarray): A batch of predicted masks of shape (batch_size, height, width), 
                                         where each pixel value represents the prediction score for that pixel.
        confidence_threshold (float): The confidence threshold above which an object is considered "found".

    Returns:
        float: The percentage of objects found with an average prediction score above the confidence threshold.
    """
    binary_masks = torch.tensor(binary_masks) if not torch.is_tensor(binary_masks) else binary_masks
    predicted_masks = torch.tensor(predicted_masks) if not torch.is_tensor(predicted_masks) else predicted_masks
    if binary_masks.ndim == 2 and predicted_masks.ndim == 2:
        predicted_masks = predicted_masks.unsqueeze(0)
        binary_masks = binary_masks.unsqueeze(0)
    binary_masks = binary_masks.cpu().numpy()
    predicted_masks = predicted_masks.cpu().numpy()
    
    total_objects = 0
    found_objects = 0
    
    batch_size = binary_masks.shape[0]
    
    for i in range(batch_size):
        binary_mask = binary_masks[i]
        predicted_mask = predicted_masks[i]
        
        labeled_mask, num_objects = label(binary_mask)
        total_objects += num_objects
        
        # Iterate over each object in the current mask
        for object_id in range(1, num_objects + 1):
            object_mask = (labeled_mask == object_id)
            avg_value = predicted_mask[object_mask].mean()
            
            # Count objects that have an average score above the confidence threshold
            if avg_value >= confidence_threshold:
                found_objects += 1
    
    # Calculate the percentage of found objects
    percentage_found = (found_objects / total_objects) * 100 if total_objects > 0 else 0
    return percentage_found


def compute_avg_object_prediction_score_by_size(binary_masks, predicted_masks,threshold=None):
    """
    Calculates the average prediction score for each object in a batch of binary masks and groups the results
    by the pixel size of the objects.

    The objects are grouped into size ranges (e.g., 0-4, 5-10 pixels), and the average score for 
    all objects in each size range is computed.

    Args:
        binary_masks (numpy.ndarray): A batch of binary masks (batch_size, height, width), where each distinct 
                                      object is represented as a connected region of 1s, and the background is 0.
        predicted_masks (numpy.ndarray): A batch of predicted score masks (batch_size, height, width), where each 
                                         pixel value represents the prediction score for that pixel.

    Returns:
        dict: A dictionary where the keys represent size ranges (e.g., '0-4', '5-10') and the values
              are the average prediction scores for objects in that size range, aggregated across the batch.
    """
    binary_masks = torch.tensor(binary_masks) if not torch.is_tensor(binary_masks) else binary_masks
    predicted_masks = torch.tensor(predicted_masks) if not torch.is_tensor(predicted_masks) else predicted_masks
    if binary_masks.ndim == 2 and predicted_masks.ndim == 2:
        predicted_masks = predicted_masks.unsqueeze(0)
        binary_masks = binary_masks.unsqueeze(0)
    binary_masks = binary_masks.cpu().numpy()
    predicted_masks = predicted_masks.cpu().numpy()
        
    # Define size ranges for grouping objects
    size_ranges = {
        '0-4': (0, 4),
        '5-10': (5, 10),
        '11-20': (11, 20),
        '21+': (21, np.inf)
    }
    
    # Create a dictionary to store the sum of scores and counts for each range
    results = defaultdict(lambda: {'sum': 0, 'count': 0})
    
    # Iterate over each mask in the batch
    batch_size = binary_masks.shape[0]
    
    for i in range(batch_size):
        binary_mask = binary_masks[i]
        predicted_mask = predicted_masks[i]
        
        # Label the distinct objects in the current binary mask
        labeled_mask, num_objects = label(binary_mask)
        
        # Iterate over each object in the current mask
        for object_id in range(1, num_objects + 1):
            # Create a mask for the current object
            object_mask = (labeled_mask == object_id)
            
            # Get the size (number of pixels) of the current object
            object_size = object_mask.sum()
            
            # Compute the average value of the predicted mask for the current object
            avg_value = predicted_mask[object_mask].mean()
            
            # Find the appropriate size range for this object
            for size_range, (min_size, max_size) in size_ranges.items():
                if min_size <= object_size <= max_size:
                    results[size_range]['sum'] += avg_value
                    results[size_range]['count'] += 1
                    break
    
    # Compute the final average scores for each size range
    avg_scores_by_size = {}
    for size_range, data in results.items():
        if data['count'] > 0:
            avg_scores_by_size[size_range] = data['sum'] / data['count']
        else:
            avg_scores_by_size[size_range] = None  # No objects in this size range

    return avg_scores_by_size


