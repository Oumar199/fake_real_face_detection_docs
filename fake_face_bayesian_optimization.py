from fake_face_detection.utils.generation import PI_generate_sample as generate_sample
from fake_face_detection.utils.acquisitions import PI_acquisition as acquisition
from fake_face_detection.utils.sampling import get_random_samples
from sklearn.gaussian_process import GaussianProcessRegressor
from functools import partial
from typing import *
import pandas as pd
import numpy as np
import string
import random
import pickle
import os

letters = string.ascii_letters + string.digits

class SimpleBayesianOptimizationForFakeReal:
    """A Bayesian Optimization class 
    """
    def __init__(self, objective: Callable, search_spaces: dict, maximize: bool = True, random_kwargs: dict = {}, kwargs: dict = {}, checkpoint: str = "data/trials/checkpoint.txt"):
        """A bayesian optimization class

        Args:
            objective (Callable): _description_
            search_spaces (dict): _description_
            maximize (bool, optional): _description_. Defaults to True.
            random_kwargs (dict, optional): _description_. Defaults to {}.
            kwargs (dict, optional): _description_. Defaults to {}.
            checkpoint (str, optional): _description_. Defaults to "data/trials/checkpoint.txt".
        """
        # recuperate the optimization strategy
        self.maximize = maximize
        
        # checkpoint where the data and score will be saved
        self.checkpoint = checkpoint
        
        # initialize the search spaces
        self.search_spaces = search_spaces
        
        # recuperate the random kwargs
        self.random_kwargs = random_kwargs
        
        # initialize the objective function
        self.objective = objective
        
        # initialize the kwargs
        self.kwargs = kwargs
        
        # initialize the model
        self.model = GaussianProcessRegressor()
        
        # initialize the random kwargs with a random values
        random_kwargs = {key: value + ''.join(random.choice(letters) for i in range(7)) for key, value in self.random_kwargs.items()}
        
        # add random kwargs to the kwargs
        self.kwargs.update(random_kwargs)
        
        # recuperate random sample
        config = get_random_samples(search_spaces)
        
        if os.path.exists(self.checkpoint):
            
            with open(self.checkpoint, 'rb') as f:
            
                pickler = pickle.Unpickler(f)
                
                checkpoint = pickler.load()
                
                self.data = checkpoint['data']
                
                self.scores = checkpoint['scores']
                
                self.model = checkpoint['model']
                
                self.current_trial = checkpoint['trial']
                
                print(f"Checkpoint loaded at trial {self.current_trial}")
        
        else:
            
            # add config to kwargs
            self.kwargs['config'] = config
            
            # calculate the first score
            score = self.objective(**self.kwargs)
            
            # initialize the input data
            self.data = [list(config.values())]
            
            # initialize the scores
            self.scores = [[score]]
            
            # fit the model with the input data and the target
            self.model.fit(self.data, self.scores)
            
            # initialize the number of trials to zero
            self.current_trial = 0
            
            with open(self.checkpoint, 'wb') as f:
                
                pickler = pickle.Pickler(f)
                
                checkpoint = {
                    'data': self.data,
                    'scores': self.scores,
                    'model': self.model,
                    'trial': self.current_trial
                }
                
                pickler.dump(checkpoint)
    
    def optimize(self, n_trials: int = 50, n_tests: int = 100):
        """Finding the best hyperparameters with the Bayesian Optimization

        Args:
            n_trials (int, optional): The number of trials. Defaults to 50.
            n_tests (int, optional): The number of random samples to test for each trial. Defaults to 100.
        """
        
        # let us make multiple trials in order to find the best params
        for trial in range(self.current_trial + 1, self.current_trial + n_trials + 1):
            
            # let us generate new samples with the acquisition and the surrogate functions
            new_sample = generate_sample(self.data, self.model, self.search_spaces, n_tests, maximize = self.maximize)
            config = {key: new_sample[i] for i, key in enumerate(self.search_spaces)}
            
            # recuperate a new score
            new_score = self.get_score(config)
            
            # let us add the new sample, target and score to their lists
            self.data.append(new_sample)
            
            self.scores.append([new_score])
            
            # let us train again the model
            self.model.fit(self.data, self.scores)
            
            # recuperate the current trial
            self.current_trial = trial
        
            with open(self.checkpoint, 'wb') as f:
                
                pickler = pickle.Pickler(f)
                
                checkpoint = {
                    'data': self.data,
                    'scores': self.scores,
                    'model': self.model,
                    'trial': self.current_trial
                }
                
                pickler.dump(checkpoint)
    
    def get_score(self, config: dict):
        
        # add random seed (since we have always the same problem of randomizing the seed)
        random.seed(None)
        
        # initialize the random kwargs with a random values
        random_kwargs = {key: value + ''.join(random.choice(letters) for i in range(7)) for key, value in self.random_kwargs.items()}
        print(random_kwargs)
        # add random kwargs to the kwargs
        self.kwargs.update(random_kwargs)
        
        # add config to kwargs
        self.kwargs['config'] = config
        
        # calculate the first score
        new_score = self.objective(**self.kwargs)
        
        return new_score
        
    def get_results(self):
        """Recuperate the generated samples and the scores

        Returns:
            pd.DataFrame: A data frame containing the results
        """
        # let us return the results as a data frame
        data = {key: np.array(self.data, dtype = object)[:, i] for i, key in enumerate(self.search_spaces)}
        
        data.update({'score': np.array(self.scores)[:, 0]})
        
        return pd.DataFrame(data)
        
        
