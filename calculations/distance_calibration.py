#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
distance_calibration.py: Contains functions related to distance calibration for
                         the Multi-Grid detector.
"""

import os
import struct
import shutil
import zipfile
import re
import numpy as np
import pandas as pd


# ==============================================================================
#                             DISTANCE CALIBRATION
# ==============================================================================

def distance_calibration(df, E_in_meV, moderator_to_sample_in_m, number_bins=100):
    """
    Function to calibrate the distance to each voxel in the MG from the sample.
    Data is collected from a monochromatic neutron beam incident on a small
    vanadium sample. The script histograms data from each voxel, takes the tof
    corrsponding to the peak maximum, and calculates the sample-detector
    distance based on this.

    Args:
        df (pd.DataFrame): DataFrame containing the Multi-Grid data
        E_in_meV (float): Incident energy of neutrons
        moderator_to_sample_in_m (float): Moderator-to-sample distance in meters
        number_bins (int): Number of bins for histogram

    Returns:
        distances (np.array): A 3D-array containing the
                              '(bus, wch, gch) -> distance'-mapping

    """

    # Define parameters
    buses = np.array([0, 1])
    whcs = np.arange(0, 80, 1)
    gchs = np.arange(80, 120, 1)
    # Calculate velocity of incident neutrons
    v_in_m_per_s = np.sqrt((E_in_meV/5.227)) * 1e3 # Squires (p. 3)
    tof_moderator_to_sample_in_s = (moderator_to_sample_in_m/v_in_m_per_s)
    # Declare 3-dimensional numpy array
    distances = np.zeros((len(buses), 80, 120), dtype='float')
    # Iterate through all voxels
    for bus in buses:
        for wch in whcs:
            for gch in gchs:
                # Get data from a single voxel
                voxel_data = df[(df.bus == bus) &
                                (df.wch == wch) &
                                (df.gch == gch)]
                # Histogram tof data
                hist, bin_edges = np.histogram(voxel_data['tof'], bins=number_bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                # Get tof corresponding to peak maximum
                idx_max_value = np.argmax(hist)
                tof_in_s = bin_centers[idx_max_value] * 1e-6
                # Get distance corresponding to tof
                tof_sample_to_detection_in_s = tof_in_s - tof_moderator_to_sample_in_s
                distance_in_m = v_in_m_per_s * tof_sample_to_detection_in_s
                # Save in distances 3D-matrix
                distances[bus, wch, gch] = distance_in_m
    return distances
