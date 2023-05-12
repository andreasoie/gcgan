import ntpath
import os
import time

import numpy as np

from . import html, util


def save_image(outdir, visuals, image_paths, aspect_ratio=1.0):
    short_path = ntpath.basename(image_paths[0])
    name = os.path.splitext(short_path)[0]
    save_path = os.path.join(outdir, f"{name}.png")
    util.save_image(visuals["fake_B"], save_path)
    
class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.opt = opt
        self.saved = False
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=opt.display_port)

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