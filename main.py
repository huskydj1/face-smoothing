# Code for server-side face-smoothing

import os
import argparse
import yaml
import time
import numpy as np

import cv2
import matplotlib
import matplotlib.pyplot as plt

from detector.detect import detect_face
from detector.smooth import smooth_face
from utils.image import (load_image, 
                         save_image, 
                         save_steps, 
                         check_img_size,
                         get_height_and_width,
                         process_image,
                         check_if_adding_bboxes)
from utils.video import (split_video,
                         process_video)
from utils.types import (is_image,
                         is_video,
                         is_directory)



def load_configs():
    """
    Loads the project configurations.

    Returns
    -------
    configs : dict
        A dictionary containing the configs
    """
    with open(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "configs/configs.yaml"
    ), 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def process(input_file, output_file):
    """Puts it all together."""
    # Start measuring time
    tic = time.perf_counter()
    # Load project configurations
    cfg = load_configs()
    # Load the network
    net = cv2.dnn.readNetFromTensorflow(cfg['net']['model_file'], 
                                        cfg['net']['cfg_file'])
    # Input and load image

    try:
        if is_image(input_file):
            # Load image
            input_img = load_image(input_file)
            print("INPUT IMAGE SHAPE", input_img.shape)

            # Process image
            img_steps = process_image(input_img, cfg, net)
            
            # Save image
            img_saved = save_image(output_file, img_steps[6])

    except ValueError:
        print('Input must be a valid image, video, or directory.')
    
    # End measuring time
    toc = time.perf_counter()
    print(f"Operation ran in {toc - tic:0.4f} seconds")

