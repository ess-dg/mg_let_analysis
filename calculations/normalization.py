#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalization.py: Contains functions related to data normalization.
"""

import os
import struct
import shutil
import zipfile
import re
import numpy as np
import pandas as pd
import scipy.optimize as sci_opt


# ==============================================================================
#                               HELIUM-3 TUBES
# ==============================================================================

def he3_get_area(number_pixels, pixels_per_tube=256):
    """
    Get area covered by the specified number of pixels in the helium-3 tubes.

    Args:
        number_pixels (np.array): Number of pixels
        pixels_per_tube (int): Number of pixels per tube

    Return:
        area_in_m2 (float): Area in m^2 covered by pixels
    """

    tube_length = 4 # [m]
    tube_diameter = 0.0254 # [m]
    segment_area = (tube_length*tube_diameter)/pixels_per_tube
    area_in_m2 = segment_area * number_pixels
    return area_in_m2

def he3_get_solid_angle(pixels, pixels_per_tube=256):
    """
    Get solid angle covered by the specified pixels in the helium-3 tubes.

    Args:
        pixels (np.array): Pixel coordinates in Helium-3 array
        pixels_per_tube (int): Number of pixels per tube

    Return:
        solid_angle (float): Solid angle in radians covered by pixels
    """

    # Get area of one pixel
    pixel_area = he3_get_area(1)
    # Get solid angle by summing contribution from each pixel
    solid_angle = 0
    for x, y, z, r in zip(pixels['x'], pixels['y'], pixels['z'], pixels['r']):
        projected_area = (pixel_area
                         * np.cos(np.arctan(np.absolute(y)/np.sqrt(x**2 + z**2))))
        pixel_solid_angle = projected_area / (r ** 2)
        solid_angle += pixel_solid_angle
    return solid_angle

def get_duration(path):
    """
    Get measurement duration of file.

    Args:
        path (str): Path to '.nxs'-file.

    Return:
        duration_in_s (float): Duration of measurements in seconds
    """

    nxs = h5py.File(path, 'r')
    duration_in_s = nxs['raw_data_1']['duration'][0] # In seconds
    return duration_in_s
