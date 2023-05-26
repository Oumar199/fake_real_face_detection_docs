from typing import *
import numpy as np
import random

def get_random_sample(search_space: dict, p: Union[List[float], None] = None):
    """Recuperate a random sample

    Args:
        search_space (dict): A dictionary defining the search space

    Raises:
        ValueError: 'min' and 'max' can only be numbers
        KeyError: Only the following keys can be provided {'min', 'max'}, {'value'}, {'values'} or {'values', 'p'} 

    Returns:
        Union[int, float, str]: The random sample 
    """
    
    keys = set(search_space)
    
    if keys == set(['min', 'max']):
        
        assert search_space['min'] < search_space['max']
        
        if isinstance(search_space['min'], int) and isinstance(search_space['max'], int):
            
            return random.randint(search_space['min'], search_space['max'])
        
        elif isinstance(search_space['min'], float) or isinstance(search_space, float):
            
            return random.uniform(search_space['min'], search_space['max'])
        
        else:
            
            raise ValueError("You can only provide int or float values with min max!")
    
    elif keys == set(['value']):
        
        return search_space['value']
    
    elif keys.issubset(set(['values'])):
        
        p = None
        
        if 'p' in keys: p = search_space['p']
        
        return np.random.choice(search_space['values'], size = (1), p = p)[0]
    
    else:
        
        raise KeyError("You didn't provide right keys! Try between: {'min', 'max'}, {'value'}, {'values'} or {'values', 'p'}")
        

def get_random_samples(search_spaces: dict):
    """Recuperate random samples from a dictionary of search spaces

    Args:
        search_spaces (dict): A dictionary where the keys are the hyperparameter names and the values are the search spaces

    Returns:
        dict: A dictionary where the keys are the hyperparameter names and the values are the sampled values from the search spaces
    """
    
    samples = {}
    
    for search_space in search_spaces:
        
        samples[search_space] = get_random_sample(search_spaces[search_space])
    
    return samples
