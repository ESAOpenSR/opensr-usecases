import numpy as np
import torch


def segmentation_metrics(binary_masks, predicted_masks, threshold=0.75):
    """
    Calculate binary segmentation metrics batch-wise.

    Args:
        binary_masks (numpy.ndarray): Ground truth binary masks (batch_size, height, width).
        predicted_masks (numpy.ndarray): Predicted masks with probability scores (batch_size, height, width).
        threshold (float): Threshold to binarize predicted masks (default is 0.5).

    Returns:
        dict: Dictionary with keys ['IoU', 'Dice', 'Precision', 'Recall', 'Accuracy'],
              each containing a list of per-image values.
    """
    
    # Binarize predicted masks based on the threshold
    predicted_binary_masks = (predicted_masks >= threshold).astype(np.uint8)
    
    # Initialize accumulators for batch metrics
    results = {
        "IoU": [],
        "Dice": [],
        "Precision": [],
        "Recall": [],
        "Accuracy": []
    }
    batch_size = binary_masks.shape[0]

    for i in range(batch_size):
        true_mask = binary_masks[i]
        pred_mask = predicted_binary_masks[i]
        
        # Calculate true positives, false positives, false negatives, true negatives - per Pixel
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

        # Append results for the current to lists
        results["IoU"].append(iou)
        results["Dice"].append(dice)
        results["Precision"].append(precision)
        results["Recall"].append(recall)
        results["Accuracy"].append(accuracy)

    return results

