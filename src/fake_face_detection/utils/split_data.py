
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
    
    
