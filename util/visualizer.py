import ntpath
import os
import time

import numpy as np

from . import html, util


def save_images(outdir, visuals, image_paths, aspect_ratio=1.0):
    """Save images to the disk.

    Parameters:
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_paths (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    short_path = ntpath.basename(image_paths[0])
    name = os.path.splitext(short_path)[0]
    
    img = visuals["fake_B"]
    save_path = os.path.join(outdir, f"{name}.png")
    util.save_image(img, save_path)