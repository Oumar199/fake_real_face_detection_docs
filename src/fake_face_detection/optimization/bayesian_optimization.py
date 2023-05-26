from fake_face_detection.utils.generation import PI_generate_sample as generate_sample
from fake_face_detection.utils.acquisitions import PI_acquisition as acquisition
from fake_face_detection.utils.sampling import get_random_samples
from sklearn.gaussian_process import GaussianProcessRegressor
from typing import *
import pandas as pd
import numpy as np

class SimpleBayesianOptimization:
    
    def __init__(self, objective: Callable, search_spaces: dict, maximize: bool = True):
        
        # recuperate the optimization strategy
        self.maximize = maximize
        
        # recuperate random sample
        sample = get_random_samples(search_spaces)
        
        # initialize the search spaces
        self.search_spaces = search_spaces
        
        # initialize the objective function
        self.objective = objective
        
        # calculate the first score
        score = objective(sample)
        
        # initialize the model
        self.model = GaussianProcessRegressor()
        
        # initialize the input data
        self.data = [list(sample.values())]
        
        # initialize the scores
        self.scores = [[score]]
        
        # fit the model with the input data and the target
        self.model.fit(self.data, self.scores)
    
    def optimize(self, n_trials: int = 50, n_tests: int = 100):
        """Finding the best hyperparameters with the Bayesian Optimization

        Args:
            n_trials (int, optional): The number of trials. Defaults to 50.
            n_tests (int, optional): The number of random samples to test for each trial. Defaults to 100.
        """
        # let us make multiple trials in order to find the best params
        for _ in range(n_trials):
            
            # let us generate new samples with the acquisition and the surrogate functions
            new_sample = generate_sample(self.data, self.model, self.search_spaces, n_tests, maximize = self.maximize)
            sample = {key: new_sample[i] for i, key in enumerate(self.search_spaces)}
            
            # let us recuperate a new score from the new sample
            new_score = self.objective(sample)
            
            # let us add the new sample, target and score to their lists
            self.data.append(new_sample)
            
            self.scores.append([new_score])
            
            # let us train again the model
            self.model.fit(self.data, self.scores)
        
    def get_results(self):
        """Recuperate the generated samples and the scores

        Returns:
            pd.DataFrame: A data frame containing the results
        """
        # let us return the results as a data frame
        data = {key: np.array(self.data, dtype = object)[:, i] for i, key in enumerate(self.search_spaces)}
        
        data.update({'score': np.array(self.scores)[:, 0]})
        
        return pd.DataFrame(data)
        
        
