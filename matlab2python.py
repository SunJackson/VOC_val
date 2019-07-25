# encoding: utf-8
import numpy as np
from PIL import Image


def matlab_imread(im_path):
    im = Image.open(im_path)  # Replace with your image name here
    indexed = np.array(im)  # Convert to NumPy array to easier access

    # Get the colour palette
    palette = im.getpalette()

    # Determine the total number of colours
    num_colours = len(palette) / 3

    # Determine maximum value of the image data type
    max_val = float(np.iinfo(indexed.dtype).max)

    # Create a colour map matrix
    map = np.array(palette).reshape(num_colours, 3) / max_val
    return  indexed, map