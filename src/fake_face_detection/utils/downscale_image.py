
from PIL.JpegImagePlugin import JpegImageFile
from PIL import Image
from math import *
import numpy as np
import itertools

def downscale_image(image: JpegImageFile, size: tuple = (224, 224)):
    
    assert image.size[0] % size[0] == 0 and image.size[1] % size[1] == 0
    
    # get box size
    height, width = int(image.size[0] / size[0]), int(image.size[1] / size[1]) 
    
    print(f"Height and width of each box: {(height, width)}")
    
    # we will concatenate the patches over the height axis (axis 0)
    patches = []
    
    for j in range(0, size[1] * width, width):
        
        # we must recuperate each width division in order to concatenate the results (on axis 1)
        h_div = []
        
        for i in range(0, size[0] * height, height):
        
            box = (j, i, j + width, i + height)
            
            current_box = image.crop(box)
            
            # let us convert the box to a numpy array and calculate the mean
            current_box = np.array(current_box).mean(axis = (0, 1))[np.newaxis, np.newaxis, :]
            
            # add to h_div
            h_div.append(current_box)
        
        # concatenate over width axis
        patches.append(np.concatenate(h_div, axis = 0))
    
    # concatenate over the height axis and transform to a pillow image
    image = Image.fromarray(np.uint8(np.concatenate(patches, axis = 1)))   
    
    return image
    
    
    
