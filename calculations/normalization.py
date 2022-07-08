#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalization.py: Contains functions related to data normalization.
"""

import numpy as np
import pandas as pd
import h5py
import scipy.optimize as sci_opt
import matplotlib.pyplot as plt

import calculations.distance_calibration as dclb


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

def he3_get_solid_angle_old(pixels, pixels_per_tube=256):
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
        projected_area = (pixel_area * np.cos(np.arctan(np.absolute(y)/np.sqrt(x**2 + z**2))))
        pixel_solid_angle = projected_area / (r ** 2)
        solid_angle += pixel_solid_angle
    return solid_angle

def he3_get_solid_angle(pixels, pixels_per_tube=256):
    """
    Get solid angle covered by the specified pixels in the helium-3 tubes.
    We assume that the tubes are always facing the sample ortohonally
    in the the (x, z)-plane.

    Args:
        pixels (np.array): Pixel coordinates in Helium-3 array
        pixels_per_tube (int): Number of pixels per tube

    Return:
        solid_angle (float): Solid angle in radians covered by pixels
    """

    # Get dimensions of one pixel
    pixel_radius = 0.0254/2 # meters
    tube_length = 4 # meters
    pixel_height = tube_length / pixels_per_tube
    # Get solid angle by summing contribution from each pixel
    phi_pairs = []
    theta_pairs = []
    solid_angle = 0
    for x_c, y_c, z_c in zip(pixels['x'], pixels['y'], pixels['z']):
        # Get limits in cartesian coordinates
        zx_angle = get_zx_angle(x_c, z_c)
        y_0, y_1 = y_c - pixel_height/2, y_c + pixel_height/2
        z_0, z_1 = z_c + pixel_radius*np.cos(zx_angle - np.pi/2), z_c + pixel_radius*np.cos(zx_angle + np.pi/2)
        x_0, x_1 = x_c + pixel_radius*np.sin(zx_angle - np.pi/2), x_c + pixel_radius*np.sin(zx_angle + np.pi/2)
        # Define limits
        phi_0, phi_1 = get_phi(x_0, y_0), get_phi(x_1, y_1)
        theta_0, theta_1 = get_theta(x_0, y_0, z_0), get_theta(x_1, y_1, z_1)
        phi_pairs.append([phi_1, phi_0])
        theta_pairs.append([theta_1, theta_0])
        # Calculate solid angle
        pixel_solid_angle = abs(phi_1 - phi_0) * abs(np.cos(theta_1) - np.cos(theta_0))
        solid_angle += pixel_solid_angle
    fig = plt.figure()
    for theta_pair, phi_pair in zip(theta_pairs, phi_pairs):
        thetas = np.array([theta_pair[0], theta_pair[0], theta_pair[1], theta_pair[1], theta_pair[0]]) * (180/np.pi)
        phis = np.array([phi_pair[0], phi_pair[1], phi_pair[1], phi_pair[0], phi_pair[0]]) * (180/np.pi)
        plt.plot(phis, thetas)
    plt.show()
    return solid_angle

def he3_get_solid_angle_new(pixels, pixels_per_tube=256):
    """
    Get solid angle covered by the specified pixels in the helium-3 tubes.

    Args:
        pixels (np.array): Pixel coordinates in Helium-3 array
        pixels_per_tube (int): Number of pixels per tube

    Return:
        solid_angle (float): Solid angle in radians covered by pixels
    """

    # Get dimensions of one pixel
    pixel_diameter = 0.0254 # meters
    tube_length = 4 # meters
    pixel_height = tube_length / pixels_per_tube
    # Get solid angle by summing contribution from each pixel
    solid_angle = 0
    for x, y, z, d in zip(pixels['x'], pixels['y'], pixels['z'], pixels['r']):
        angle = np.arctan(np.absolute(y)/np.sqrt(x**2 + z**2))
        y_p= pixel_height * np.cos(angle)
        xz_p = pixel_diameter
        solid_angle_pixel = 4*np.arcsin((y_p * xz_p)/np.sqrt((y_p**2 + 4*d**2) * (xz_p**2 + 4*d**2)))
        solid_angle += solid_angle_pixel
    return solid_angle

def mg_get_solid_angle(gchs, offset, mg_theta):
    # Declare parameters
    wchs = np.array([0, 16, 32, 48, 64, 80])
    voxel_length_half = 0.025/2
    # Iterate through surface voxels
    solid_angle = 0
    zs_pairs = []
    xs_pairs = []
    ys_pairs = []
    phi_pairs = []
    theta_pairs = []
    area = 0
    for gch in gchs:
        for wch in wchs:
            # Get limits in cartesian coordinates
            x_c, y_c, z_c = dclb.get_global_xyz(wch, gch, offset, mg_theta)
            y_0, y_1 = y_c - voxel_length_half, y_c + voxel_length_half
            z_0, z_1 = z_c + voxel_length_half*np.cos(mg_theta), z_c + voxel_length_half*np.cos(mg_theta - np.pi)
            x_0, x_1 = x_c + voxel_length_half*np.sin(mg_theta), x_c + voxel_length_half*np.sin(mg_theta - np.pi)
            
            zs_pairs.append([z_0, z_1])
            xs_pairs.append([x_0, x_1])
            # Define limits
            phi_0, phi_1 = get_phi(x_0, y_0), get_phi(x_1, y_1)
            theta_0, theta_1 = get_theta(x_0, y_0, z_0), get_theta(x_1, y_1, z_1)
            phi_pairs.append([phi_1, phi_0])
            theta_pairs.append([theta_1, theta_0])
            # Calculate solid angle
            voxel_solid_angle = abs(phi_1 - phi_0) * abs(np.cos(theta_1) - np.cos(theta_0))
            area += np.sqrt((z_1 - z_0)**2 + (x_1 - x_0)**2) * abs((y_1 - y_0))
            solid_angle += voxel_solid_angle
    fig = plt.figure()
    for theta_pair, phi_pair in zip(theta_pairs, phi_pairs):
        thetas = np.array([theta_pair[0], theta_pair[0], theta_pair[1], theta_pair[1], theta_pair[0]]) * (180/np.pi)
        phis = np.array([phi_pair[0], phi_pair[1], phi_pair[1], phi_pair[0], phi_pair[0]]) * (180/np.pi)
        plt.plot(phis, thetas)
    #plt.xlim(-160, -125)
    #plt.ylim(30, 50)
    plt.show()
    #print('Area', area)
    #fig = plt.figure()
    #for xs_pair, zs_pair in zip(xs_pairs, zs_pairs):
    #    plt.plot(zs_pair, xs_pair)
    #plt.gca().set_aspect('equal')
    #plt.show()
    return solid_angle

def mg_get_solid_angle_new(gchs, offset, mg_theta):
    # Declare parameters
    wchs = np.array([0, 16, 32, 48, 64, 80])
    voxel_length = 0.025
    # Iterate through surface voxels
    solid_angle = 0
    for gch in gchs:
        for wch in wchs:
            # Get limits in cartesian coordinates
            x, y, z = dclb.get_global_xyz(wch, gch, offset, mg_theta)
            d = np.sqrt(x**2 + y**2 + z**2)
            angle_1 = np.arctan(np.absolute(y)/np.sqrt(x**2 + z**2))
            angle_2 = get_zx_angle(x, z) + np.pi/2
            y_p= voxel_length * np.cos(angle_1)
            xz_p = voxel_length * np.cos(angle_2 - mg_theta)
            solid_angle_voxel = 4*np.arcsin((y_p * xz_p)/np.sqrt((y_p**2 + 4*d**2) * (xz_p**2 + 4*d**2)))
            solid_angle += solid_angle_voxel
    return solid_angle

def he3_get_duration(path):
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

def he3_get_start_time(path):
    """
    Get measurement duration of file.

    Args:
        path (str): Path to '.nxs'-file.

    Return:
        start_tim
    """

    nxs = h5py.File(path, 'r')
    start_time = str(nxs['raw_data_1']['start_time'][0])
    return start_time


# ==============================================================================
#                             MULTI-GRID DETECTOR
# ==============================================================================

def mg_get_solid_angle_old(gchs, offset, theta):
    # Declare parameters
    wchs = np.array([0, 16, 32, 48, 64, 80])
    voxel_length_xz = 0.025
    voxel_length_y = 0.025
    # Iterate through surface voxels
    solid_angle = 0
    for gch in gchs:
        for wch in wchs:
            x, y, z = dclb.get_global_xyz(wch, gch, offset, theta)
            r = np.sqrt(x**2 + y**2 + z**2)
            angle = np.arctan(np.absolute(y)/np.sqrt(x**2 + z**2))
            voxel_length_y_projected = voxel_length_y * np.cos(angle)
            projected_area = voxel_length_y_projected * voxel_length_xz
            voxel_solid_angle = projected_area / (r ** 2)
            solid_angle += voxel_solid_angle
    return solid_angle


            

# ==============================================================================
#                           HELPER FUNCTIONS
# ==============================================================================
            
def get_zx_angle(x, z):
    partial_angle = np.arctan(abs(x)/(abs(z)))
    if (z > 0) and (x > 0):
        full_angle = partial_angle
    elif (z < 0) and (x > 0):
        full_angle = np.pi/2 + np.pi/2 - partial_angle
    elif (z < 0) and (x < 0):
        full_angle = np.pi + partial_angle
    else:
        full_angle = 2*np.pi - partial_angle
    return full_angle

def get_phi(x, y):
    partial_angle = np.arctan(y/x)
    if x > 0:
        phi = partial_angle
    if (x < 0) and (y >= 0):
        phi = partial_angle + np.pi
    elif (x < 0) and (y < 0):
        phi = partial_angle - np.pi
    elif (x == 0) and (y > 0):
        phi = np.pi/2
    elif (x == 0) and (y < 0):
        phi = -np.pi/2
    elif (x == 0) and (y == 0):
        phi = None
    return phi

def get_theta(x, y, z):
    theta = np.arctan(np.sqrt(x**2 + y**2)/z)
    return theta
    
    
