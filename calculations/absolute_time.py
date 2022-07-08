#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalization.py: Contains functions related to data normalization.
"""

import numpy as np
import pandas as pd
import h5py
import scipy.optimize as sci_opt

import calculations.distance_calibration as dclb


# ==============================================================================
#                               HELIUM-3 TUBES
# ==============================================================================

def he3_get_start_time_in_posix(path):
    """
    Get measurement duration of file.

    Args:
        path (str): Path to '.nxs'-file.

    Return:
        start_tim
    """

    nxs = h5py.File(path, 'r')
    start_time = str(nxs['raw_data_1']['start_time'][0])
    print(start_time)
    # Extract start time in posix
    year = int(start_time[2:6])
    print('Year', year)
    month = int(start_time[7:9])
    print('Month', month)
    print(start_time[10:12])
    day = int(start_time[11:13])
    print('Day', day)
    print(start_time[13:15])
    hour = int(start_time[13:15])
    print('Hour', hour)
    minute = int(start_time[11:13])
    second = int(start_time[13:15])
    print('Year', year)
    print('Month', month)
    print('Day', day)
    print('...')
    print('Hour', hour)
    print('Minute', minute)
    print('Second', second)
    print('...')
    dt = datetime.datetime(year, month, day, hour, minute, second)
    print(dt)
    print('...')
    time_in_posix = datetime.datetime.timestamp(dt)
    return start_time


