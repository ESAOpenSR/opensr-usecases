import torch
import numpy as np

def min_max_stretch(x, min_val=0.0, max_val=1.0):
    """
    Stretches input x to the range [min_val, max_val].
    Supports numpy arrays and torch tensors.
    """
    try:
        is_torch = isinstance(x, torch.Tensor)
    except ImportError:
        is_torch = False

    if is_torch:
        x_min = x.amin()
        x_max = x.amax()
        stretched = (x - x_min) / (x_max - x_min + 1e-8)
        return stretched * (max_val - min_val) + min_val
    else:
        x = np.asarray(x)
        x_min = x.min()
        x_max = x.max()
        stretched = (x - x_min) / (x_max - x_min + 1e-8)
        return stretched * (max_val - min_val) + min_val