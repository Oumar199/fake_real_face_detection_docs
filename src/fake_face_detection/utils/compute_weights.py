import sklearn.utils as skl
from typing import *
import numpy as np

def compute_weights(samples: List[int]):
    """Compute the weights with the 'balanced' method

    Args:
        samples (List[int]): The samples: A list of integers

    Returns:
        numpy.ndarray: A array containing the weights
    """
    
    # get unique classes
    classes = np.unique(samples)
    
    # calculate the weights with the balanced method
    weights = skl.class_weight.compute_class_weight('balanced', classes=classes, y = samples)
    
    return weights
