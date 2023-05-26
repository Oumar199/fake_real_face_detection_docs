
from PIL.JpegImagePlugin import JpegImageFile
from math import *
import itertools

def get_patches(image: JpegImageFile, n_patches: int):
    
    # get height and width of the image
    height, width = image.size 
    
    # let us calculate the number of divisions to make to the width and height of the image
    n_patch = int(sqrt(n_patches)) 

    patch_h = int(height / n_patch) # notice that the height must be divisible by the number of divisions

    patch_w = int(width / n_patch) # notice that the width must be divisible by the number of divisions

    print(f"Height and width of each patch: {(patch_h, patch_w)}")
    
    # we will find the first coordinates of the boxes with product function of itertools
    first_coordinates = list(itertools.product(range(0, patch_h * n_patch, patch_h),
                                        range(0, patch_w * n_patch, patch_w)))
    
    patches = []
    
    for pos1, pos2 in first_coordinates:
        
        box = (pos2, pos1, pos2 + patch_w, pos1 + patch_h)
        
        patches.append(image.crop(box))
    
    return patches
    
    
    
