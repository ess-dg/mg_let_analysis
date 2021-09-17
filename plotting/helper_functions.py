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
