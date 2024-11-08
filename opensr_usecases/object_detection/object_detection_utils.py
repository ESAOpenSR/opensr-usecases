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

def standard_metrics(binary_masks, predicted_masks, threshold=0.5):
    """
    Calculate binary segmentation metrics batch-wise.

    Args:
        binary_masks (numpy.ndarray): Ground truth binary masks (batch_size, height, width).
        predicted_masks (numpy.ndarray): Predicted masks with probability scores (batch_size, height, width).
        threshold (float): Threshold to binarize predicted masks (default is 0.5).

    Returns:
        dict: A dictionary containing batch-wise metrics (IoU, Dice, precision, recall, accuracy).
    """
    
    # Binarize predicted masks based on the threshold
    predicted_binary_masks = (predicted_masks >= threshold).astype(np.uint8)
    
    # Initialize accumulators for batch metrics
    iou_accum = 0
    dice_accum = 0
    precision_accum = 0
    recall_accum = 0
    accuracy_accum = 0
    batch_size = binary_masks.shape[0]
    
    for i in range(batch_size):
        true_mask = binary_masks[i]
        pred_mask = predicted_binary_masks[i]
        
        # Calculate true positives, false positives, false negatives, true negatives
        tp = np.sum((true_mask == 1) & (pred_mask == 1))
        fp = np.sum((true_mask == 0) & (pred_mask == 1))
        fn = np.sum((true_mask == 1) & (pred_mask == 0))
        tn = np.sum((true_mask == 0) & (pred_mask == 0))
        
        # Intersection over Union (IoU)
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
        
        # Accumulate metrics for each batch
        iou_accum += iou
        dice_accum += dice
        precision_accum += precision
        recall_accum += recall
        accuracy_accum += accuracy
    
    # Compute the average of metrics across the batch
    metrics = {
        'IoU': iou_accum / batch_size,
        'Dice': dice_accum / batch_size,
        'Precision': precision_accum / batch_size,
        'Recall': recall_accum / batch_size,
        'Accuracy': accuracy_accum / batch_size
    }
    
    return metrics



