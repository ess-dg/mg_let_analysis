#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
energy.py: Contains functions which makes energy calculations.
"""

import os
import struct
import shutil
import zipfile
import re
import numpy as np
import pandas as pd


# ==============================================================================
#                                    ENERGY
# ==============================================================================

def get_energy(tof_in_us, sample_to_detection_in_m, moderator_to_sample_in_m):
    """
    Function to calculate neutron energy by first calculating neutron velocity
    from time-of-flight and flight distance.

    Args:
        tof_in_us (np.array): Time-of-flight in micro seconds
        sample_to_detection_in_m (np.array): Sample-to-detection distance in
                                             meters
        moderator_to_sample_in_m (float): Moderator-to-sample distance in meters

    Return:
        energies_in_meV (np.array): Neutron energies in meV
    """

    # Energy calculation
    full_distance = moderator_to_sample_in_m + sample_to_detection_in_m
    v_in_km_per_s = (full_distance * 1e-3)/(tof_in_us * 1e-6)
    energies_in_meV = 5.227 * (v_in_km_per_s ** 2) # Squires (p. 3)
    return energies_in_meV


# ==============================================================================
#                                 ENERGY TRANSFER
# ==============================================================================

def get_energy_transfer(Ei_in_meV, tof_in_us, sample_to_detection_in_m,
                        moderator_to_sample_in_m):
    """
    Function to calculate neutron energy transfer by calculating the initial
    neutron velocity, v_i, as well as the final neutron velocity, v_f,
    after the scattering.

    Args:
        Ei_in_meV (float): Initial energy in meV
        tof_in_us (np.array): Time-of-flight in micro seconds
        sample_to_detection_in_m (np.array): Sample-to-detection distance
                                             in meters
        moderator_to_sample_in_m (float): Moderator-to-sample distance in meters

    Return:
        delta_E (np.array): Neutron energy transfer (Ei - Ef)
    """

    # Get initial neutron velocity
    vi_in_m_per_s = np.sqrt((Ei_in_meV/5.227)) * 1e3 # Squires (p. 3)
    # Get final neutron velocity
    tof_moderator_to_sample_in_s = (moderator_to_sample_in_m/vi_in_m_per_s)
    tof_sample_to_detection_in_s = (tof_in_us * 1e-6) - tof_moderator_to_sample_in_s
    vf_in_km_per_s = (sample_to_detection_in_m/tof_sample_to_detection_in_s) * 1e-3
    # Get final neutron energy
    Ef_in_meV = 5.227 * (vf_in_km_per_s ** 2) # Squires (p. 3)
    # Calculate energy transfer
    delta_E = Ei_in_meV - Ef_in_meV
    return delta_E


# ==============================================================================
#                          WAVELENGTH-ENERGY CONVERTION
# ==============================================================================

def meV_to_A(energy):
    """ Convert energy in meV to wavelength in Angstrom."""
    return np.sqrt(81.81/energy)

def A_to_meV(wavelength):
    """ Convert wavelength in Angstrom to energy in meV."""
    return (81.81/(wavelength ** 2))
