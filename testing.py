from data.dataset_usa_buildings import SegmentationDataset
ds = SegmentationDataset(phase="train",image_type="sr")
im,mask = ds.__getitem__(10)
from collections import defaultdict



import numpy as np
from scipy.ndimage import label

def compute_avg_object_prediction_score(binary_mask, predicted_mask):
    """
    Calculates the average prediction score for each object in a binary mask.

    This function identifies individual objects in a binary mask by labeling connected components. 
    For each labeled object, it calculates the average value of the corresponding region in the 
    predicted mask.

    Args:
        binary_mask (numpy.ndarray): A binary mask where each distinct object is represented 
                                     as a connected region of 1s, and the background is 0.
        predicted_mask (numpy.ndarray): A mask of predicted scores, where each pixel value represents 
                                        the prediction score for that pixel.

    Returns:
        list: A list of average prediction scores for each object in the binary mask. The length of 
              the list corresponds to the number of distinct objects in the binary mask.
    """

    
    labeled_mask, num_objects = label(binary_mask)
    
    avg_values = []
    
    # Iterate over each object
    for object_id in range(1, num_objects + 1):
        # Create a mask for the current object
        object_mask = (labeled_mask == object_id)
        
        # Compute the average value of the predicted mask for the current object
        avg_value = predicted_mask[object_mask].mean()
        avg_values.append(avg_value)
    
    return avg_values


def compute_avg_object_prediction_score_by_size(binary_masks, predicted_masks):
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
    if binary_masks.ndim == 2 and predicted_masks.ndim == 2:
        predicted_masks = predicted_masks.unsqueeze(0)
        binary_masks = binary_masks.unsqueeze(0)
        
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

compute_avg_object_prediction_score(mask,mask)

compute_avg_object_prediction_score_by_size(mask,mask)