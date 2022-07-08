#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
helper_functions.py: Contains the helper functions for plotting
"""
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from os import makedirs,path

# =============================================================================
#                             CUSTOMIZE THICK LABELS
# =============================================================================

def set_thick_labels(thickness):
    # Customize matplotlib font sizes
    plt.rc('font', size=thickness)          # controls default text sizes
    plt.rc('axes', titlesize=thickness)     # fontsize of the axes title
    plt.rc('axes', labelsize=thickness)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=thickness)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=thickness)    # fontsize of the tick labels
    plt.rc('legend', fontsize=thickness)    # legend fontsize
    plt.rc('figure', titlesize=thickness)   # fontsize of the figure title
    
def mkdir_p(my_path):
    """
    Creates a directory, equivalent to using mkdir -p on the command line.

    Args:
        my_path (str): Path to where the new folder should be created.

    Yields:
        A new folder at the requested path.
    """
    try:
        makedirs(my_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(my_path):
            pass
        else: raise
