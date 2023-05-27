# Usage 

## Installation

The `fake_face_detection` package contains functions and classes used for making exploration, pre-processing, visualization, training, searching for the best model, etc. It is available in the following GitHub repository [Fake_face_detection](https://github.com/Oumar199/fake_face_detection_ViT), and you install it with the following steps:

- Type the following command on the console to clone the GitHub repository:
```console
$ git clone https://github.com/Oumar199/fake_face_detection_ViT.git
```
- Enter the cloned directory with the command:
```console
$ cd fake_face_detection_ViT
```
- Create a python environment with `virtualenv`:
```console
$ pip install virtualenv
$ python<version> -m venv <virtual-environment-name>
```
- Install the required libraries in your environment by typing the following command:
```console
$ pip install -r requirements.txt
```
- Finally install the package with:
```console
$ pip install -e fake-face-detection
```

## Example

After installing the package, you can test it by creating a Python file named <i style="color:orange">optimization</i> and add the following code inside the file to optimize the parameters of your objective function:
```python
# import the Bayesian optimization class
from fake_face_detection.optimization.bayesian_optimization import SimpleBayesianOptimization
import pandas as pd

"""
Create here your objective function and define your search spaces according to the Tutorial
"""

# Initialize the Bayesian optimization object
bo_search = SimpleBayesianOptimization(objective, search_spaces) # if you want to minimize the objective function set maximize = False

# Search for the best hyperparameters
bo_search.optimize(n_trials = 50, n_tests = 100)

# Print the results
results = bo_search.get_results()

pd.options.display.max_rows = 50
print(results.head(50))

```

To execute the file, write the following command in the console of your terminal:
```console
$ python<version> optimization
```




