#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fitting.py: Contains functions related to fitting procedures of the data.
"""

import numpy as np
import pandas as pd
import scipy.optimize as sci_opt


# ==============================================================================
#                                GAUSSIAN FIT
# ==============================================================================

def fit_data(hist, bins):
    """
    Function to fit a Gaussian distribution.

    Args:
        hist (np.array): Histogram
        bins (np.array): Bin centers

    Return:
        a (float): a, as defined below
        x0 (float): x0, as defined below
        sigma (float): sigma, as defined below
        perr (np.array): Array with uncertainites on fit parameters
    """

    a_guess, x0_guess, sigma_guess = get_fit_parameters_guesses(hist, bins)
    popt, pcov = sci_opt.curve_fit(Gaussian, bins, hist,
                                   p0=[a_guess, x0_guess, sigma_guess])
    a, x0, sigma = popt[0], popt[1], abs(popt[2])
    perr = np.sqrt(np.diag(pcov))
    return a, x0, sigma, perr


# ==============================================================================
#                           GAUSSIAN PARAMETER GUESSES
# ==============================================================================

def get_fit_parameters_guesses(hist, bins):
    """
    Function to estimate the parameters of a Gaussian.

    Args:
        hist (np.array): Histogram
        bins (np.array): Bin centers

    Return:
        a_guess (float): Guess of a
        x0_guess (float): Guess of x0
        sigma_guess (float): Guess of sigma
    """

    # Extract relavant parameters
    maximum = max(hist)
    maximum_idx = find_nearest(hist, maximum)
    half_maximum = maximum/2
    half_maximum_idx_1 = find_nearest(hist[:maximum_idx], half_maximum)
    half_maximum_idx_2 = find_nearest(hist[maximum_idx:], half_maximum) + maximum_idx
    FWHM = bins[half_maximum_idx_2] - bins[half_maximum_idx_1]
    # Calculate guesses
    a_guess = maximum
    x0_guess = bins[maximum_idx]
    sigma_guess = FWHM/(2*np.sqrt(2*np.log(2)))
    return a_guess, x0_guess, sigma_guess



# ==============================================================================
#                                  LINEAR FIT
# ==============================================================================

def fit_linear(hist, bins):
    """
    Function to fit a Linear distribution.

    Args:
        hist (np.array): Histogram
        bins (np.array): Bin centers

    Return:
        k (float): Slope
        m (float): Constant
        perr (np.array): Array with uncertainites on fit parameters
    """
    popt, pcov = sci_opt.curve_fit(Linear, bins, hist)
    k, m = popt[0], popt[1]
    perr = np.sqrt(np.diag(pcov))
    return k, m, perr


# ==============================================================================
#                            HELPER FUNCTIONS
# ==============================================================================

def Gaussian(x, a, x0, sigma):
    """ Calculates the value from a gaussian function."""
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def Linear(x, k, m):
    """ Calculates the value from a linear function."""
    return x*k + m

def find_nearest(array, value):
    """ Returns closest element in array to value."""
    idx = (np.abs(array - value)).argmin()
    return idx
