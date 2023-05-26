from torch.utils.tensorboard  import SummaryWriter
from PIL.JpegImagePlugin import JpegImageFile
import matplotlib.pyplot as plt
from typing import *
from math import *
import numpy as np
import random
import torch
import os

# use a style with no grid
plt.style.use("_mpl-gallery-nogrid")

def visualize_images(images_dict: Dict[str, Iterable[Union[JpegImageFile, torch.Tensor, np.ndarray]]], 
                     log_directory: str = "fake_face_logs",
                     n_images: int = 40,
                     figsize = (15, 15), 
                     seed: Union[int, None] = None):
    
    assert len(images_dict) > 0
    
    assert isinstance(images_dict, dict) 
    
    # add seed
    random.seed(seed)
    
    # verify if we must add a title for each image
    add_titles = len(images_dict) > 1
    
    images_ = []
    
    # modify the dictionary to obtain a tuple of images with their corresponding tags
    for key in images_dict:
        
        for image in images_dict[key]:
            
            images_.append((key, image))
        
    # we take the number of images in the list if n_images is larger
    if n_images > len(images_): n_images = len(images_)
    
    # choose random images
    images = random.choices(images_, k = n_images)
    
    if isinstance(images[0], JpegImageFile):
        
        images = [np.array(image[1]) for image in images if type(image[1]) in [JpegImageFile, torch.Tensor, np.ndarray]]
    
    # calculate the number of rows and columns
    n_rows = ceil(sqrt(n_images))
    
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_rows, figsize = figsize)
    
    # flat the axes
    axs = axs.flat
    
    # trace images
    for i in range(n_images):
        
        axs[i].imshow(images[i][1], interpolation = "nearest")
        
        if add_titles: axs[i].set_title(images[i][0])
        
        axs[i].axis('off')
    
    # add padding to the figure
    fig.tight_layout()
    
    # deleting no necessary plots
    [fig.delaxes(axs[i]) for i in range(n_images, n_rows * n_rows)]
    
    # add figure to tensorboard
    with SummaryWriter(os.path.join(log_directory, "images")) as writer:
        
        # identify the tag
        tag = "_".join(list(images_dict)) if add_titles else list(images_dict.keys())[0]
        
        writer.add_figure(tag = tag, figure = fig)
    
