from fake_face_detection.utils.acquisitions import PI_acquisition
from fake_face_detection.utils.sampling import get_random_samples
from sklearn.gaussian_process import GaussianProcessRegressor
from typing import *
import numpy as np

def PI_generate_sample(X: Iterable, model: GaussianProcessRegressor, search_spaces: dict, n_tests: int = 100, maximize: bool = True):
    """Generate new samples with the probability of improvement

    Args:
        X (Iterable): The list of input data
        model (GaussianProcessRegressor): The model to train
        search_spaces (dict): The search spaces
        n_tests (int, optional): The number of random samples to test. Defaults to 100.
        maximize (bool, optional): The optimization strategy. If maximize == True -> maximize, else -> minimize. Defaults to True.

    Returns:
        List: The new sample
    """
    
    # let us create random samples
    X_prime = [list(get_random_samples(search_spaces).values()) for i in range(n_tests)]
    
    # let us recuperate the probabilities from the acquisition function
    probs = PI_acquisition(X, X_prime, model, maximize = maximize)
    
    # let us return the best sample
    return X_prime[np.argmax(probs)]
