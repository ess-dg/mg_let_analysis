#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
distance_calibration.py: Contains functions related to distance calibration for
                         the Multi-Grid detector.
"""

import numpy as np
import pandas as pd
import sys


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
    whcs = np.arange(0, 96, 1)
    gchs = np.arange(96, 133, 1)
    # Calculate velocity of incident neutrons
    v_in_m_per_s = np.sqrt((E_in_meV/5.227)) * 1e3 # Squires (p. 3)
    tof_moderator_to_sample_in_s = (moderator_to_sample_in_m/v_in_m_per_s)
    # Declare 2-dimensional numpy array
    distances = np.zeros((96, 37), dtype='float')
    # Iterate through all voxels
    for wch in whcs:
        for gch in gchs:
            # Get data from a single voxel
            voxel_data = df[(df.wch == wch) & (df.gch == gch)]
            # Histogram tof data
            hist, bin_edges = np.histogram(voxel_data['tof'], bins=number_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            # Get tof corresponding to peak maximum
            idx_max_value = np.argmax(hist)
            tof_in_s = bin_centers[idx_max_value] * 62.5e-9
            # Get distance corresponding to tof
            tof_sample_to_detection_in_s = tof_in_s - tof_moderator_to_sample_in_s
            distance_in_m = v_in_m_per_s * tof_sample_to_detection_in_s
            # Save in distances 3D-matrix
            distances[wch, gch-96] = distance_in_m
    return distances

# ==============================================================================
#                                GET XYZ-COORDINATES
# ==============================================================================

def get_local_xyz(wch, gch):
    x = (235.96 - 10*(wch % 16)) * 1e-3
    y = (-80.76 - 25*(37 - (gch - 96))) * 1e-3
    z = (-217.5 + 25*(wch // 16)) * 1e-3
    return x, y, z

def get_global_xyz(wch, gch, offset, theta):
    # Get coordinate in local coordinate system
    x, y, z = get_local_xyz(wch, gch)
    # Rotate according to rotation
    xr, zr = get_new_x(x, z, theta), get_new_z(x, z, theta)
    # Translate into position in global coordinate system
    xrt, yt, zrt = xr+offset['x'], y+offset['y'], zr+offset['z']
    return xrt, yt, zrt
    
def get_new_z(x, z, theta):
    partial_angle = np.arctan(abs(x)/(abs(z)))
    if (z >= 0) and (x >= 0):
        full_angle = partial_angle
    elif (z <= 0) and (x >= 0):
        full_angle = np.pi/2 + np.pi/2 - partial_angle
    elif (z <= 0) and (x <= 0):
        full_angle = np.pi + partial_angle
    elif (z >= 0) and (x <= 0):
        full_angle = 2*np.pi - partial_angle
    return np.cos(full_angle+theta)*np.sqrt(x**2 + z**2)

def get_new_x(x, z, theta):
    partial_angle = np.arctan(abs(x)/(abs(z)))
    if (z >= 0) and (x >= 0):
        full_angle = partial_angle
    elif (z <= 0) and (x >= 0):
        full_angle = np.pi - partial_angle
    elif (z <= 0) and (x <= 0):
        full_angle = np.pi + partial_angle
    elif (z >= 0) and (x <= 0):
        full_angle = 2*np.pi - partial_angle
    return np.sin(full_angle+theta)*np.sqrt(x**2 + z**2)


# ==============================================================================
#                            GET DISTANCES
# ==============================================================================

def get_sample_to_detection_distances(wchs, gchs, offset, theta):
    # Get '(wch, gch)->distance'-mapping
    distance_mapping = np.zeros((96, 37), dtype='float')
    for wch in range(0, 96):
        for gch in range(96, 133):
            x, y, z = get_global_xyz(wch, gch, offset, theta)
            distance_in_m = np.sqrt(x**2 + y**2 + z**2)
            distance_mapping[wch, gch-96] = distance_in_m
    # Get all distances
    distances = distance_mapping[wchs, gchs-96]
    return distances