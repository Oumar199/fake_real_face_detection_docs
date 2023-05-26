Bayesian Optimization from Scratch 🔝
--------------------------

We must create some functions in this notebook that will help us make Bayesian Optimization to search for the best hyperparameter that will give the best model on the validation set. Bayesian Optimization uses Gaussian Process to approximate the objective function given input data. In our case, the input data is the set of hyperparameter values we must tune. The Bayesian theorem is used to direct the search to find the minimum or maximum of our objective function. The objective function can be the `Accuracy,` the `Recall,` or whatever score function to evaluate the model performance. 

The Bayesian Theorem suggests identifying a prior distribution for the objective function and updating it with the likelihood obtained with data given the objective function to get a posterior distribution. The Gaussian Process (GP) is commonly used for noisy distribution, which is difficult to directly simple from. The GP doesn't need to optimize parameters to find the distribution and can give a standard derivation around the mean distribution, which defines the uncertainty about the approximation. 

The prior distribution is aMultivariate normal distribution with mean $m_0$ and variance $K_{x, x}$, which is commonly a kernel distribution like the Radial Basis function (RBF) and calculates the similarity between the input values. Some noise $\epsilon$ are added to the Prior Distribution with a mean of 0 and a variance $\sigma^2$. The posterior distribution is also Multivariate normal distribution for which we are searching the mean $m_y$ vector and the covariance matrix $\Sigma$. The `Cholesty iterative method` is commonly used to find the best posterior mean and covariance for a given new point. We will explain further the Gaussian Process. 

For now, let us focus on Bayesian Optimization: 

- After finding the posterior distribution, we can obtain the mean value of the objective function for any group of new hyperparameters. The new objective function values are used to find the best new samples for the subsequent evaluation trial since we make trials before finding the best hyperparameter values.

- The first trial finds the first score from the objective function given random hyperparameter values. 

- A `surrogate function` is used to give that score from the Posterior Multivariate Distribution of the objective function given the input data.

- The estimated score from the `surrogate function` is then used in a new function named `acquisition function` to generate new samples from the hyperparameter's search spaces. It exists many different `acquisition functions.`

- The new samples are concatenated to the previous ones and used to train a further Posterior distribution. 

- The process is repeated until finding the most satisfying score.

**Note**:  This idea is related to reinforcement learning methods to search for the following action(s) which maximize the value function (the reward of long term). We can consider the value function to be the cumulative distribution function of the approximate Posterior Multivariate Distribution function over the new samples and the new actions to be the ones sampled from the value function plus a variance rate to explore more states. The states are abstracted (not visible) in our case.

We will implement the Bayesian Optimization process from scratch since we want to customize it for the current project. We will not code the Gaussian Process Regression sub-process since it is already integrated into the `scikit-learn` library. We can also use the `GPytorch` library which can provide better result but for the purpose of that project we will not need it. Let us define nextly the objective function. 

The following package is installed after finishing to implement the functions in the current notebook.


```python
!pip install -e fake-face-detection -qq
```

Let us import the necessary libraries.


```python
from sklearn.gaussian_process import GaussianProcessRegressor
from functools import partial
from torch.nn import MSELoss
from scipy.stats import norm
from functools import partial
from typing import *
from torch import nn
import pandas as pd
import numpy as np
import random
import torch
```

    c:\Users\Oumar Kane\AppData\Local\pypoetry\Cache\virtualenvs\pytorch1-HleOW5am-py3.10\lib\site-packages\tqdm\auto.py:22: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    

### Objective function

We will use the `MSELoss` to calculate the value returned by the objective function, which is our training function. It will calculate the mean squared error between values predicted from a feed-forward neural network and pre-defined target values. The input and target values are randomly initialized from a Gaussian distribution. We will need input data of 8 variables and 100 samples (not very large, not fast up the training). The following parameters will be necessary for a sample example:

- The number of epochs -> ... [1, 10]
- The number of layers -> ... [1, 4]
- The number of features -> ... [40, 100]
- The learning rate -> ... [1e-1, 1e-4]

Let us initialize the input data and the targets.


```python
X = torch.randn((100, 8))

y = torch.rand((100, 1))
```

Let us initialize the model and the objective function. Notice that we must add the noise we defined earlier to the final calculated score. The noise is sampled from a normal distribution with a mean of 0 and a scale that we must determine. Let us choose a $\sigma^2 = 0.1$ scale as the default.


```python
# model
def set_model(input_size: int = 8, n_features: int = 1, n_layers: int = 1):
    
    layers = [nn.Sequential(nn.Linear(input_size, n_features), nn.ReLU())]
    
    layers.extend([nn.Sequential(nn.Linear(n_features, n_features), nn.ReLU()) for i in range(n_layers - 1)])
    
    layers.append(nn.Sequential(nn.Linear(n_features, 1)))
    
    sequence = nn.Sequential(*layers)
    
    return sequence

# Only one iteration will be sufficient
def objective(optimizer: nn.Module, loss_fn: nn.Module, input: torch.Tensor, target: torch.Tensor, params: dict, scale: float = 0.1):
    
    noise = torch.distributions.normal.Normal(0.0, scale).sample().item()
    
    model = set_model(n_features=params['n_features'], n_layers=params['n_layers'])
    
    optimizer_ = optimizer(model.parameters(), lr = params['lr'])
    
    losses = []
    
    for _ in range(params['epochs']):
        
        outputs = model(input)
        
        loss = loss_fn(outputs, target)
        
        loss.backward()
        
        optimizer_.step()
        
        optimizer_.zero_grad()
        
        losses.append(loss.item())
    
    return 1 / np.mean(losses) + noise
```

We must also define simple functions which generate random samples from search spaces.


```python
%%writefile fake-face-detection/fake_face_detection/utils/sampling.py
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
```

    Overwriting fake-face-detection/fake_face_detection/utils/sampling.py
    


```python
%run fake-face-detection/fake_face_detection/utils/sampling.py
```

Let us test the sampling functions and do training to obtain a first value.


```python
# define the search spaces
search_spaces = {'epochs': {
    'min': 1,
    'max': 10
    },
    'n_layers': {
        'values': [1, 2, 3, 4]
    },
    'n_features': {
        'value': 50
    },
    'lr': {
        'min': 1e-4,
        'max': 1e-1
    }
}

# recuperate random samples
samples = get_random_samples(search_spaces)
```

We obtain the following samples for each hyperparameter.


```python
samples
```




    {'epochs': 10, 'n_layers': 4, 'n_features': 50, 'lr': 0.010000772954938224}



Let us train the model with them and recuperate the loss.


```python
loss = objective(torch.optim.Adam, nn.MSELoss(), X, y, samples)
```

We obtain the following loss from the sampled hyperparameter values.


```python
loss
```




    7.645348566937997



Let us now implement the surrogate function

### The surrogate function

The surrogate tries to estimate the objective function using the Bayes theorem probabilistically. We want to find the probability of obtaining such a score $f$ conditionally to input data $D$. The score is the loss calculated after training, and the input data is the set of parameters. To simulate the posterior distribution $P(f/D)$, we decided to use the Gaussian Process (GP) Regression. The kernel used to calculate the similarity between the input data sample can be the `Radial Basis Function` kernel which commonly provides excellent results. The GP Regression is already implemented in sklearn. We can use it directly for our implementation.


```python
# define the distribution
gp_model = GaussianProcessRegressor()
```

The only data we have is the sample hyperparameter value which instantiates the input data.


```python
# instantiate the input data
data = [list(samples.values())]
```

And the only score we currently have is the loss we calculated earlier.


```python
# instantiate the scores
scores = [[loss]]
```

Let us fit the model with the data and scores.


```python
gp_model.fit(data, scores)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GaussianProcessRegressor()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">GaussianProcessRegressor</label><div class="sk-toggleable__content"><pre>GaussianProcessRegressor()</pre></div></div></div></div></div>



We can estimate the value of the target using the input data (and the standard deviation).


```python
pred, stds = gp_model.predict(data, return_std=True)
```

We obtain the following prediction, which is very close to the actual loss.


```python
pred, stds
```




    (array([7.64534857]), array([1.00000004e-05]))



It is time to choose an acquisition function to generate new samples.

### Acquisition Function

The acquisition function will use the surrogate to examine which of many random samples is the best suited for the next generation:

1. First, we need to generate random samples from the search spaces.
2. Second, since we already have an approximation of the objective function's distribution conditionally to a input data via the surrogate function we can estimate the vector mean and their corresponding standard deviations:
$$\mu_e, \sigma_e \sim P(f/samples)
$$ (where $b$ is the number random samples) and $estimated\_stds obtained from the samples generated at 1 and the corresponding standard deviations.
3. Third, the values in the vector of the estimated means, , are compared to the best mean: 
$$\max(\mu), \space where \space \mu,. \sim P(f/input\_data)$$ 
Calculated from the previous best input data calculating the probability of improvement which is the cumulative normal distribution of the normalized distance between the means (or the regret as in RL):

$$
PI = P(f < \frac{\mu_e - \mu_{best}}{\sigma_e + \epsilon})
$$

Where $\epsilon$ is added to avoid division by zero.

**Remark**: it exists different acquisition functions like the UCB function. But for the purpose of our project we will focus on the probability of improvement.

**Note**: In our example we want to minimize the loss so we must take $-\mu_e$ and $-\mu$ to find the best solution.


Let us implement bellow the acquisition function.


```python
%%writefile fake-face-detection/fake_face_detection/utils/acquisitions.py
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
```

    Overwriting fake-face-detection/fake_face_detection/utils/acquisitions.py
    


```python
%run fake-face-detection/fake_face_detection/utils/acquisitions.py
```

The next generated sample is the one which have the best probability of being chosen.


```python
%%writefile fake-face-detection/fake_face_detection/utils/generation.py
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
```

    Overwriting fake-face-detection/fake_face_detection/utils/generation.py
    


```python
%run fake-face-detection/fake_face_detection/utils/generation.py
```

Let us generate the next sample with the above function.


```python
new_samples = PI_generate_sample(data, gp_model, search_spaces, maximize = False)
```

We obtained the following new values for the next training.


```python
new_samples
```




    [6, 1, 50, 0.04335985788917143]



Let us train again the model and recuperate the new score.


```python
# initialize the new dictionary of samples
params = {key: new_samples[i] for i, key in enumerate(search_spaces)}

# calculate the new score
new_score = objective(torch.optim.Adam, nn.MSELoss(), X, y, params)

```


```python
new_score
```




    4.486408321363162



We don't have enough data so we can obtain a worth loss. We must concatenate the generated samples and scores in order to obtain a more accurate prediction from the surrogate function.


```python
data.append(new_samples)
scores.append([new_score])
```


```python
%%writefile fake-face-detection/fake_face_detection/optimization/bayesian_optimization.py
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
        
        
```

    Overwriting fake-face-detection/fake_face_detection/optimization/bayesian_optimization.py
    


```python
%run fake-face-detection/fake_face_detection/optimization/bayesian_optimization.py
```

Let us train recuperate the samples obtained after 50 trials.


```python
# initialize the attributes
simple_bayesian_optimization = SimpleBayesianOptimization(partial(objective, torch.optim.Adam, nn.MSELoss(), X, y), search_spaces, maximize=False)

# optimize to find the best hyperparameters
simple_bayesian_optimization.optimize(50)

# recuperate the results
results = simple_bayesian_optimization.get_results()
```


```python
# display the results
pd.options.display.max_rows = 50
results.head(50)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>epochs</th>
      <th>n_layers</th>
      <th>n_features</th>
      <th>lr</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>2</td>
      <td>50</td>
      <td>0.068535</td>
      <td>0.650897</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>50</td>
      <td>0.062454</td>
      <td>2.223563</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>4</td>
      <td>50</td>
      <td>0.028624</td>
      <td>6.153221</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>50</td>
      <td>0.026828</td>
      <td>4.910913</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>4</td>
      <td>50</td>
      <td>0.065757</td>
      <td>0.319705</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8</td>
      <td>1</td>
      <td>50</td>
      <td>0.008961</td>
      <td>7.682987</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>4</td>
      <td>50</td>
      <td>0.062921</td>
      <td>0.209620</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>4</td>
      <td>50</td>
      <td>0.014572</td>
      <td>6.000742</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4</td>
      <td>4</td>
      <td>50</td>
      <td>0.04908</td>
      <td>0.393076</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3</td>
      <td>4</td>
      <td>50</td>
      <td>0.091362</td>
      <td>0.162732</td>
    </tr>
    <tr>
      <th>10</th>
      <td>9</td>
      <td>4</td>
      <td>50</td>
      <td>0.073419</td>
      <td>0.211130</td>
    </tr>
    <tr>
      <th>11</th>
      <td>10</td>
      <td>3</td>
      <td>50</td>
      <td>0.028732</td>
      <td>5.491310</td>
    </tr>
    <tr>
      <th>12</th>
      <td>10</td>
      <td>1</td>
      <td>50</td>
      <td>0.078995</td>
      <td>3.034438</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3</td>
      <td>3</td>
      <td>50</td>
      <td>0.09489</td>
      <td>-0.034298</td>
    </tr>
    <tr>
      <th>14</th>
      <td>4</td>
      <td>3</td>
      <td>50</td>
      <td>0.099335</td>
      <td>-0.019501</td>
    </tr>
    <tr>
      <th>15</th>
      <td>4</td>
      <td>3</td>
      <td>50</td>
      <td>0.029398</td>
      <td>4.716212</td>
    </tr>
    <tr>
      <th>16</th>
      <td>4</td>
      <td>4</td>
      <td>50</td>
      <td>0.089937</td>
      <td>-0.028549</td>
    </tr>
    <tr>
      <th>17</th>
      <td>8</td>
      <td>4</td>
      <td>50</td>
      <td>0.044093</td>
      <td>0.789361</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2</td>
      <td>1</td>
      <td>50</td>
      <td>0.053627</td>
      <td>2.606744</td>
    </tr>
    <tr>
      <th>19</th>
      <td>6</td>
      <td>1</td>
      <td>50</td>
      <td>0.060509</td>
      <td>3.968602</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
      <td>1</td>
      <td>50</td>
      <td>0.016641</td>
      <td>1.942502</td>
    </tr>
    <tr>
      <th>21</th>
      <td>8</td>
      <td>3</td>
      <td>50</td>
      <td>0.055132</td>
      <td>1.049847</td>
    </tr>
    <tr>
      <th>22</th>
      <td>3</td>
      <td>2</td>
      <td>50</td>
      <td>0.085914</td>
      <td>0.224853</td>
    </tr>
    <tr>
      <th>23</th>
      <td>7</td>
      <td>3</td>
      <td>50</td>
      <td>0.090876</td>
      <td>0.113944</td>
    </tr>
    <tr>
      <th>24</th>
      <td>6</td>
      <td>2</td>
      <td>50</td>
      <td>0.092498</td>
      <td>0.332945</td>
    </tr>
    <tr>
      <th>25</th>
      <td>4</td>
      <td>4</td>
      <td>50</td>
      <td>0.096601</td>
      <td>0.109976</td>
    </tr>
    <tr>
      <th>26</th>
      <td>7</td>
      <td>2</td>
      <td>50</td>
      <td>0.042129</td>
      <td>2.791446</td>
    </tr>
    <tr>
      <th>27</th>
      <td>7</td>
      <td>3</td>
      <td>50</td>
      <td>0.004674</td>
      <td>4.443855</td>
    </tr>
    <tr>
      <th>28</th>
      <td>4</td>
      <td>4</td>
      <td>50</td>
      <td>0.087317</td>
      <td>0.214934</td>
    </tr>
    <tr>
      <th>29</th>
      <td>3</td>
      <td>3</td>
      <td>50</td>
      <td>0.095541</td>
      <td>-0.037780</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2</td>
      <td>2</td>
      <td>50</td>
      <td>0.002686</td>
      <td>3.725293</td>
    </tr>
    <tr>
      <th>31</th>
      <td>8</td>
      <td>4</td>
      <td>50</td>
      <td>0.086471</td>
      <td>0.176205</td>
    </tr>
    <tr>
      <th>32</th>
      <td>3</td>
      <td>3</td>
      <td>50</td>
      <td>0.098342</td>
      <td>-0.118936</td>
    </tr>
    <tr>
      <th>33</th>
      <td>8</td>
      <td>3</td>
      <td>50</td>
      <td>0.097012</td>
      <td>0.134272</td>
    </tr>
    <tr>
      <th>34</th>
      <td>7</td>
      <td>3</td>
      <td>50</td>
      <td>0.095638</td>
      <td>0.207632</td>
    </tr>
    <tr>
      <th>35</th>
      <td>4</td>
      <td>2</td>
      <td>50</td>
      <td>0.092748</td>
      <td>0.253815</td>
    </tr>
    <tr>
      <th>36</th>
      <td>5</td>
      <td>2</td>
      <td>50</td>
      <td>0.093047</td>
      <td>0.321280</td>
    </tr>
    <tr>
      <th>37</th>
      <td>9</td>
      <td>2</td>
      <td>50</td>
      <td>0.063862</td>
      <td>1.601892</td>
    </tr>
    <tr>
      <th>38</th>
      <td>5</td>
      <td>3</td>
      <td>50</td>
      <td>0.084194</td>
      <td>0.101736</td>
    </tr>
    <tr>
      <th>39</th>
      <td>5</td>
      <td>3</td>
      <td>50</td>
      <td>0.099682</td>
      <td>-0.099425</td>
    </tr>
    <tr>
      <th>40</th>
      <td>6</td>
      <td>3</td>
      <td>50</td>
      <td>0.088712</td>
      <td>0.028726</td>
    </tr>
    <tr>
      <th>41</th>
      <td>3</td>
      <td>3</td>
      <td>50</td>
      <td>0.099056</td>
      <td>-0.005610</td>
    </tr>
    <tr>
      <th>42</th>
      <td>1</td>
      <td>3</td>
      <td>50</td>
      <td>0.08012</td>
      <td>2.755868</td>
    </tr>
    <tr>
      <th>43</th>
      <td>6</td>
      <td>3</td>
      <td>50</td>
      <td>0.071228</td>
      <td>0.153810</td>
    </tr>
    <tr>
      <th>44</th>
      <td>5</td>
      <td>4</td>
      <td>50</td>
      <td>0.021338</td>
      <td>3.632742</td>
    </tr>
    <tr>
      <th>45</th>
      <td>3</td>
      <td>1</td>
      <td>50</td>
      <td>0.09476</td>
      <td>1.481126</td>
    </tr>
    <tr>
      <th>46</th>
      <td>7</td>
      <td>4</td>
      <td>50</td>
      <td>0.079477</td>
      <td>0.231139</td>
    </tr>
    <tr>
      <th>47</th>
      <td>9</td>
      <td>3</td>
      <td>50</td>
      <td>0.07009</td>
      <td>0.177824</td>
    </tr>
    <tr>
      <th>48</th>
      <td>9</td>
      <td>3</td>
      <td>50</td>
      <td>0.099022</td>
      <td>-0.049942</td>
    </tr>
    <tr>
      <th>49</th>
      <td>1</td>
      <td>4</td>
      <td>50</td>
      <td>0.053302</td>
      <td>2.900642</td>
    </tr>
  </tbody>
</table>
</div>



Let us print the best loss and the corresponding hyperparameters.


```python
results[results['score'] == results['score'].min()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>epochs</th>
      <th>n_layers</th>
      <th>n_features</th>
      <th>lr</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32</th>
      <td>3</td>
      <td>3</td>
      <td>50</td>
      <td>0.098342</td>
      <td>-0.118936</td>
    </tr>
  </tbody>
</table>
</div>



Let us get the loss with best parameters without adding a large noise.


```python
# recuperate the parameters
params = results[results['score'] == results['score'].min()]

params = params.drop('score', axis = 1).to_dict('list')

params = {key: value[0] for key, value in params.items()}

# train and get the loss
objective(torch.optim.Adam, nn.MSELoss(), X, y, params, scale=1e-5)
```




    0.013614560035266577



We highly progressed since the first random sample ✌️.
