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




def compute_found_objects_percentage(gt_mask, pred_mask, confidence_threshold=0.5):
    """
    Calculates the percentage of ground truth objects that are considered 'found' based on the predicted mask.

    An object is considered 'found' if the average predicted score within its region is above the confidence threshold.

    Args:
        gt_mask (np.ndarray): Binary ground truth mask of shape (H, W) where objects are 1 and background is 0.
        pred_mask (np.ndarray): Predicted score mask of shape (H, W) with values in [0, 1].
        confidence_threshold (float): Threshold above which an object is considered found.

    Returns:
        float: Percentage of objects found (0–100).
    """
    if gt_mask.ndim != 2 or pred_mask.ndim != 2:
        raise ValueError("gt_mask and pred_mask must be 2D arrays")

    labeled_mask, num_objects = label(gt_mask)
    if num_objects == 0:
        return 0.0

    found_objects = 0
    for object_id in range(1, num_objects + 1):
        object_region = labeled_mask == object_id
        #avg_score = pred_mask[object_region].mean()
        if (pred_mask[object_region] >= confidence_threshold).any():
            found_objects += 1

    return (found_objects / num_objects) * 100



def compute_avg_object_prediction_score_by_size(binary_masks, predicted_masks,size_ranges, threshold=None):
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



def compute_found_objects_percentage_by_size(gt_mask, pred_mask, size_ranges, threshold=0.75):
    """
    Computes the percentage of ground-truth objects detected, grouped by size bin.

    An object is considered found if at least one pixel in the predicted mask (after thresholding) overlaps with it.

    Args:
        gt_mask (np.ndarray): Binary ground truth mask (H, W).
        pred_mask (np.ndarray): Predicted soft mask (H, W), values in [0,1].
        size_ranges (dict): Dict like {'0-4': (0,4), '5-10': (5,10), ...}.
        threshold (float): Threshold to binarize predicted mask.

    Returns:
        dict: {size_bin: percent_found}, values in [0, 100].
    """
    pred_bin = pred_mask >= threshold
    labeled_gt, num_objects = label(gt_mask)

    found_counts = defaultdict(int)
    total_counts = defaultdict(int)

    for object_id in range(1, num_objects + 1):
        object_mask = (labeled_gt == object_id)
        size = object_mask.sum()

        # Find the bin this object belongs to
        bin_name = None
        for name, (min_size, max_size) in size_ranges.items():
            if min_size <= size <= max_size:
                bin_name = name
                break
        if bin_name is None:
            continue  # skip if size falls outside all bins

        total_counts[bin_name] += 1
        #avg_score = pred_mask[object_mask].mean()
        if (pred_mask[object_mask] >= threshold).any():
            found_counts[bin_name] += 1

    result = {}
    for bin_name in size_ranges.keys():
        total = total_counts[bin_name]
        found = found_counts[bin_name]
        result[bin_name] = (found / total * 100) if total > 0 else None

    return result



def compute_object_detection_per_instance(gt_mask, pred_mask):
    """
    For each object in the GT mask, check if it overlaps with any predicted object.

    Args:
        gt_mask (np.ndarray): Ground truth binary mask.
        pred_mask (np.ndarray): Predicted binary mask (same shape).

    Returns:
        List of tuples: [(size_in_pixels, matched_bool), ...]
    """
    labeled_gt, num_gt = label(gt_mask)
    detected = []

    for obj_id in range(1, num_gt + 1):
        obj_mask = (labeled_gt == obj_id)
        obj_size = np.sum(obj_mask)
        matched = np.any(pred_mask[obj_mask])
        detected.append((obj_size, matched))

    return detected
