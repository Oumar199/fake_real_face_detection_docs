from PIL.JpegImagePlugin import JpegImageFile
from PIL import ImageDraw
from PIL import Image
from typing import *

def display(images: List[JpegImageFile], labels: List[str], w: int = 300, h: int = 200, left_color: str = "white", right_color: str = "white"):
    """Display a dual image

    Args:
        images (List[JpegImageFile]): A list containing two images
        labels (List[str]): The labels of the images
        w (int, optional): The width. Defaults to 300.
        h (int, optional): The height. Defaults to 200.
        left_color (str, optional): The color of left label. Defaults to "white".
        right_color (str, optional): The color of the right label. Defaults to "white".

    Returns:
        PIL.Image: A pillow image
    """
    
    # define a grid
    grid = Image.new('RGB', size=(w, h))
    
    # draw the grid
    draw = ImageDraw.Draw(grid, mode='RGB')
    
    # define the second box
    box = (w // 2,  0)
    
    # define the size of the images
    size = (w // 2, h)
    
    # add images to the grid
    grid.paste(images[0].resize(size))
    
    grid.paste(images[1].resize(size), box = box)
    
    # draw labels
    draw.text((0, 0), labels[0], fill=left_color)
    
    draw.text(box, labels[1], fill=right_color)
    
    return grid
