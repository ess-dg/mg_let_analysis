#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bm_read.py: Reads data collected with LET beam monitors
"""

import numpy as np
import pandas as pd
import h5py

# ==============================================================================
#                           IMPORT BEAM MONITOR DATA
# ==============================================================================

def import_data(path):
    """ Imports data from LET beam monitors.

    Args:
        path (str): Path to '.nxs'-file that contains the data

    Returns:
        df (DataFrame):
    """
    # Import data
    nxs = h5py.File(path, 'r')
    # Declare keys in dictionary
    tof_edges = nxs['raw_data_1']['monitor_1']['time_of_flight'][()]
    tof_centers = (tof_edges[:-1] + tof_edges[1:]) / 2
    bm_dict = {key: [] for key in tof_centers}
    for i in np.arange(1, 9, 1):
        monitor = 'monitor_%d' % i
        histogram = nxs['raw_data_1'][monitor]['data'][()][0][0]
        for tof_center, counts in zip(tof_centers, histogram):
            bm_dict[tof_center].append(counts)
    # Initialize DataFrame
    df = pd.DataFrame(bm_dict)
    return df
