Split data 🫧
----------------------------
In this notebook, we will split the dataset between training, validation and test sets. It is part of the model selection. The training set will be used to train the model and must be at least $50\%$ of the whole dataset in order to make the model to make the difference between a real image and a photoshopped one. The number of training images must tend to infinity in order to find the right pattern or model that most fits the images and also to not over-fit. The validation set is used to select the best model or set of hyperparameters and also the evaluate the model during the training. The test is used only to verify if the model is generalized on non seen images. We will the stratified random sampling in order to obtain the proportion of labels in each set. That is we randomly sample by strate where each strate is a group of images sharing the same label. 

Since we have only $2041$ images and it is a small size, we decided to take $80\%$ of the dataset to train the model, $10\%$ to validate the model and $10\%$ to test. We will take the same proportions that indicated in the first pie chart of the following image:

![train_test_split](https://assets-global.website-files.com/5d7b77b063a9066d83e1209c/613ec5b6c3da5313e1abcc47_UeKfm9v6E9QobwFfG3ud_20Q82QoqI8W6kXQnDm_QBnOVyQXCNmwjWtMI5vD9du4cjovnpzSYBbIDHdSU-57H1Bb4DfuUCaSjZjozKIwD0IQsH7FyMuFTW7aYVW-zelk2RNMAez3%3Ds0.png)

Let us create a function which splits the dataset and create new directories for them.


```python
%%writefile fake-face-detection/fake_face_detection/utils/split_data.py

from sklearn.model_selection import train_test_split
from glob import glob
import shutil
import os

def split_data_from_dir(path: str, new_path: str, test_size: float = 0.2, valid_size: float = 0.2, force_placement: bool = True):
    
    assert test_size > 0 and test_size < 0.5 and valid_size >= 0 and valid_size < 0.5
    
    assert os.path.exists(path) and os.path.isdir(path)
    
    assert os.path.exists(new_path) and os.path.isdir(new_path)
    
    # let us recuperate the images' path from the directory
    dirs = os.listdir(path)
    
    # let us recuperate the image of each directory and split the images before making them in new directories
    for dir_ in dirs:
        
        # let us recuperate the path of the directory
        dir_path = os.path.join(path, dir_)
        
        # let us verify if it is truly a directory before making the following processes
        if os.path.isdir(dir_path):
            
            # let us recuperate the files' paths in it
            images = os.listdir(dir_path)
            
            # let us split the data between training and test set
            train_set, test_set = train_test_split(images, test_size = test_size)
            
            # let us split the training set between training and validation set
            train_set, valid_set = train_test_split(train_set, test_size = valid_size)
            
            # let us create the train test and valid directories
            if not os.path.exists(os.path.join(os.path.join(new_path, 'train'), dir_)) or\
                not os.path.exists(os.path.join(os.path.join(new_path, 'test'), dir_)) or\
                    not os.path.exists(os.path.join(os.path.join(new_path, 'valid'), dir_)):
                        
                        [os.makedirs(os.path.join(os.path.join(new_path, set_), dir_)) for set_ in ['train', 'test', 'valid']]
            
            elif not force_placement:
                
                raise OSError(f"One of the training, validation or testing directory for the class {dir_} already exists! Enable the force_placement argument if you want to use already created directories.")
                
            # let us place the sets in their locations
            [shutil.copyfile(os.path.join(dir_path, image), os.path.join(os.path.join(os.path.join(new_path, 'train'), dir_), image)) for image in train_set]
            [shutil.copyfile(os.path.join(dir_path, image), os.path.join(os.path.join(os.path.join(new_path, 'test'), dir_), image)) for image in test_set]
            [shutil.copyfile(os.path.join(dir_path, image), os.path.join(os.path.join(os.path.join(new_path, 'valid'), dir_), image)) for image in valid_set]
            
    print(f"All the file in {path} was copied in {new_path} successfully!")
    
    
```

    Overwriting fake-face-detection/fake_face_detection/utils/split_data.py
    

Let us create the training, validation and test sets.


```python
%run fake-face-detection/fake_face_detection/utils/split_data.py

split_data_from_dir('data/real_and_fake_face/', 'data/real_and_fake_splits/', test_size = 0.1,
                    valid_size = 0.1)
```

    All the file in data/real_and_fake_face/ was copied in data/real_and_fake_splits/ successfully!
    
