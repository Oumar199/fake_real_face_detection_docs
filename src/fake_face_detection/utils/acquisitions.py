from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
from typing import *

def PI_acquisition(X: List, X_prime: List, model: GaussianProcessRegressor, maximize: bool = True):
    """Acquisition function for bayesian optimization using probability of improvement

    Args:
        X (List): A list containing the input data
        X_prime (List): A list containing the generate samples
        model (GaussianProcessRegressor): The gaussian model to use
        maximize (bool, optional): A boolean value indicating the optimization objective. Defaults to True.

    Returns:
        List: A list containing the probabilities
    """
    
    # let us predict the means for the input data
    mu = model.predict(X)
    
    # let us calculate the means and standard deviation for the random samples
    mu_e, std_e = model.predict(X_prime, return_std=True)
    
    if not maximize:
        
        mu = -mu
        
        mu_e = -mu_e
    
    # let us take the best mean
    mu_best = max(mu)
    
    # let us calculate the probability of improvement
    probs = norm.cdf((mu_e - mu_best) / std_e)
    
    return probs
