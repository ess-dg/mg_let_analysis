#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lineshape.py: Contains the functions related to lineshape analysis
"""
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd

import plotting.common_plot as cmplt

import calculations.fitting as fit

# ==============================================================================
#                              GET FIGURE-OF-MERIT
# ==============================================================================

def get_figure_of_merit(Ei_in_meV, delta_E, hist, bin_edges, label=''):
    # Get bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Fit data
    be_s = 6
    be_e = 8
    a, x0, sigma, perr = fit.fit_data(hist, bin_centers)
    print('--------------------------------------------------')
    print('FITTING RESULTS - %s, %.2f meV' % (label, Ei_in_meV))
    print('--------------------------------------------------')
    print('Gaussian (y=a*exp(-(x-x0)^2/(2*σ^2)))')
    print('a  = %f ± %f' % (a, perr[0]))
    print('x0 = %f ± %f' % (x0, perr[1]))
    print('σ  = %f ± %f' % (sigma, perr[2]))
    print()
    
    # Plot Gaussian peak fit
    min_val, max_val = - 0.2 * Ei_in_meV, 0.2 * Ei_in_meV
    xx = np.linspace(min_val, max_val, 1000)
    plt.plot(xx, fit.Gaussian(xx, a, x0, sigma), color='red', zorder=5,
             label='Gaussian fit')

    # Plot deliminaters
    plt.vlines(3*sigma, 1e-3, 1e8, color='blue', label='3σ')
    plt.vlines(5*sigma, 1e-3, 1e8, color='green', label='±5σ')
    plt.vlines(-5*sigma, 1e-3, 1e8, color='green', label=None)
    plt.vlines(be_s*sigma, 1e-3, 1e8, color='orange', label='Background')
    plt.vlines(-be_s*sigma, 1e-3, 1e8, color='orange', label=None)
    plt.vlines(be_e*sigma, 1e-3, 1e8, color='orange', label=None)
    plt.vlines(-be_e*sigma, 1e-3, 1e8, color='orange', label=None)
    
    # Get linear background fit
    left_start = fit.find_nearest(bin_centers, -be_e*sigma)
    left_end = fit.find_nearest(bin_centers, -be_s*sigma)
    right_start = fit.find_nearest(bin_centers, be_s*sigma)
    right_end = fit.find_nearest(bin_centers, be_e*sigma)
    background_hist = np.concatenate((hist[left_start:left_end],
                                      hist[right_start:right_end]))
    background_bins = np.concatenate((bin_centers[left_start:left_end],
                                      bin_centers[right_start:right_end]))
    
    # Plot linear fit
    k, m, ferr = fit.fit_linear(background_hist, background_bins)
    print('Linear (y = k*x + m)')
    print('k  = %f ± %f' % (k, ferr[0]))
    print('m  = %f ± %f' % (m, ferr[1]))
    plt.plot(xx, fit.Linear(xx, k, m), color='purple', zorder=5,
             label='Linear\nbackground fit')
    plt.legend(loc=2)

    # Define edges
    left_idx_peak = fit.find_nearest(bin_centers, -5*sigma)
    right_idx_peak = fit.find_nearest(bin_centers, 5*sigma)
    left_idx_shoulder = fit.find_nearest(bin_centers, 3*sigma)
    right_idx_shoulder = fit.find_nearest(bin_centers, 5*sigma)

    # Get background counts
    counts_back_peak = sum(fit.Linear(bin_centers[left_idx_peak:right_idx_peak], k, m))
    counts_back_shoulder = sum(fit.Linear(bin_centers[left_idx_shoulder:right_idx_shoulder], k, m))

    # Get total counts
    counts_signal_and_back_peak = len(delta_E[(delta_E > -5*sigma) & (delta_E < 5*sigma)])
    counts_signal_and_back_shoulder = len(delta_E[(delta_E > 3*sigma) & (delta_E < 5*sigma)])

    # Get signal
    counts_signal_peak = counts_signal_and_back_peak - counts_back_peak
    counts_signal_shoulder = counts_signal_and_back_shoulder - counts_back_shoulder

    # Store shoulder areas, peaks areas, foms and incident energies
    fom_value = counts_signal_shoulder/counts_signal_peak
    peak_counts = counts_signal_peak
    shoulder_counts = counts_signal_shoulder
    incident_energy = Ei_in_meV

    # Cross-checking
    print()
    print('--------------------------------------------------')
    print('STATISTICS')
    print('--------------------------------------------------')
    print('Shoulder')
    print('Total: %.2f counts' % counts_signal_and_back_shoulder)
    print('Background: %.2f counts' % counts_back_shoulder)
    print('Signal: %.2f counts' % counts_signal_shoulder)
    print()
    print('Peak')
    print('Total: %.2f counts' % counts_signal_and_back_peak)
    print('Background: %.2f counts' % counts_back_peak)
    print('Signal: %.2f counts' % counts_signal_peak)
    print('---')
    
    return fom_value, peak_counts, shoulder_counts
    
def get_figure_of_merit_discrete(Ei_in_meV, delta_E, hist, bin_edges, factor=0.1, label='', discrete=False):
    def get_fom(sigma, be_s, be_e, delta_E):
        # Calulate sum in peak
        counts_signal_and_back_peak = len(delta_E[(delta_E > -5*sigma) & (delta_E < 5*sigma)])
        counts_signal_and_back_shoulder = len(delta_E[(delta_E > 3*sigma) & (delta_E < 5*sigma)])
        # Calculate average background rate
        background_sum = len(delta_E[((delta_E > be_s*sigma) & (delta_E < be_e*sigma)) | 
                                     ((delta_E < -be_s*sigma) & (delta_E > -be_e*sigma))
                                    ])
        dE_region = 2*(be_e - be_s)*sigma
        background_rate_per_dE = background_sum/dE_region
        # Estimate background underneath shoulder and peak
        counts_back_peak = background_rate_per_dE * (10*sigma)
        counts_back_shoulder = background_rate_per_dE * (2*sigma)
        # Get signal
        counts_signal_peak = counts_signal_and_back_peak - counts_back_peak
        counts_signal_shoulder = counts_signal_and_back_shoulder - counts_back_shoulder
        # Store shoulder areas, peaks areas, foms and incident energies
        fom_value = counts_signal_shoulder/counts_signal_peak
        peak_counts = counts_signal_peak
        shoulder_counts = counts_signal_shoulder
        incident_energy = Ei_in_meV
        return fom_value, peak_counts, shoulder_counts, incident_energy, background_rate_per_dE
    
    def get_debugging_values(sigma, be_s, be_e, delta_E):
        # Calulate sum in peak
        counts_signal_and_back_peak = len(delta_E[(delta_E > -5*sigma) & (delta_E < 5*sigma)])
        counts_signal_and_back_shoulder = len(delta_E[(delta_E > 3*sigma) & (delta_E < 5*sigma)])
        # Calculate average background rate
        background_sum = len(delta_E[((delta_E > be_s*sigma) & (delta_E < be_e*sigma)) | 
                                     ((delta_E < -be_s*sigma) & (delta_E > -be_e*sigma))
                                    ])
        dE_region = 2*(be_e - be_s)*sigma
        background_rate_per_dE = background_sum/dE_region
        # Estimate background underneath shoulder and peak
        counts_back_peak = background_rate_per_dE * (10*sigma)
        counts_back_shoulder = background_rate_per_dE * (2*sigma)
        # Get signal
        counts_signal_peak = counts_signal_and_back_peak - counts_back_peak
        counts_signal_shoulder = counts_signal_and_back_shoulder - counts_back_shoulder
        # Store shoulder areas, peaks areas, foms and incident energies
        fom_value = counts_signal_shoulder/counts_signal_peak
        peak_counts = counts_signal_peak
        shoulder_counts = counts_signal_shoulder
        incident_energy = Ei_in_meV
        return counts_signal_and_back_peak, counts_signal_and_back_shoulder, counts_back_peak, counts_back_shoulder, counts_signal_peak, counts_signal_shoulder, shoulder_counts, peak_counts

    # Define background region
    be_s = 11 # 12
    be_e = 15 # 17
    min_val, max_val = - 0.2 * Ei_in_meV, 0.2 * Ei_in_meV
    #factor = 0.05
    if discrete:
        # Extract data of interest
        min_dE, max_dE = - Ei_in_meV*factor, Ei_in_meV*factor
        data = delta_E[(delta_E >= min_dE) & (delta_E <= max_dE)]
        # Get sigma
        mu = (1/len(data)) * sum(data)
        sigma_discrete = np.sqrt((1/len(data)) * sum((data - mu)**2))
        # Get fom
        fom_value, peak_counts, shoulder_counts, incident_energy, background_rate_per_dE = get_fom(sigma_discrete, be_s, be_e, delta_E)
        # Get debugging values
        counts_signal_and_back_peak, counts_signal_and_back_shoulder, counts_back_peak, counts_back_shoulder, counts_signal_peak, counts_signal_shoulder, shoulder_counts, peak_counts = get_debugging_values(sigma_discrete, be_s, be_e, delta_E)
        # Calculate statistical errors
        sigma_discrete_err = sigma_discrete/np.sqrt(len(data))
        sigma_discrete_min = sigma_discrete - sigma_discrete_err
        sigma_discrete_max = sigma_discrete + sigma_discrete_err
        # Get min-fom and max-fom
        fom_value_min, __, __, __, __ = get_fom(sigma_discrete_min, be_s, be_e, delta_E)
        fom_value_max, __, __, __, __ = get_fom(sigma_discrete_max, be_s, be_e, delta_E)
        print('Sigmas', sigma_discrete, sigma_discrete_err, sigma_discrete_min, sigma_discrete_max)
        print('foms', fom_value, fom_value_min, fom_value_max)
        # Plot deliminaters
        plt.vlines(min_dE, 1e-3, 1e8, color='black', label=None)
        plt.vlines(max_dE, 1e-3, 1e8, color='black', label=None)
        plt.vlines(3*sigma_discrete, 1e-3, 1e8, color='blue', label='3σ')
        plt.vlines(5*sigma_discrete, 1e-3, 1e8, color='green', label='±5σ')
        plt.vlines(-5*sigma_discrete, 1e-3, 1e8, color='green', label=None)
        plt.vlines(be_s*sigma_discrete, 1e-3, 1e8, color='orange', label='Background')
        plt.vlines(-be_s*sigma_discrete, 1e-3, 1e8, color='orange', label=None)
        plt.vlines(be_e*sigma_discrete, 1e-3, 1e8, color='orange', label=None)
        plt.vlines(-be_e*sigma_discrete, 1e-3, 1e8, color='orange', label=None)
        plt.hlines(background_rate_per_dE*(bin_edges[1]-bin_edges[0]), min_val, max_val, color='purple', label='Average background')
        plt.legend(loc=2)
    else:
        # Get bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # Fit data
        a, x0, sigma, perr = fit.fit_data(hist, bin_centers)
        print('--------------------------------------------------')
        print('FITTING RESULTS - %s, %.2f meV' % (label, Ei_in_meV))
        print('--------------------------------------------------')
        print('Gaussian (y=a*exp(-(x-x0)^2/(2*σ^2)))')
        print('a  = %f ± %f' % (a, perr[0]))
        print('x0 = %f ± %f' % (x0, perr[1]))
        print('σ  = %f ± %f' % (sigma, perr[2]))
        print('Sigma from discrete calculation: %f' % sigma_discrete)
        print()

        # Plot Gaussian peak fit
        min_val, max_val = - 0.2 * Ei_in_meV, 0.2 * Ei_in_meV
        xx = np.linspace(min_val, max_val, 1000)
        plt.plot(xx, fit.Gaussian(xx, a, x0, sigma), color='red', zorder=5,
                 label='Gaussian fit')

        # Plot deliminaters
        plt.vlines(3*sigma, 1e-3, 1e8, color='blue', label='3σ')
        plt.vlines(5*sigma, 1e-3, 1e8, color='green', label='±5σ')
        plt.vlines(-5*sigma, 1e-3, 1e8, color='green', label=None)
        plt.vlines(be_s*sigma, 1e-3, 1e8, color='orange', label='Background')
        plt.vlines(-be_s*sigma, 1e-3, 1e8, color='orange', label=None)
        plt.vlines(be_e*sigma, 1e-3, 1e8, color='orange', label=None)
        plt.vlines(-be_e*sigma, 1e-3, 1e8, color='orange', label=None)

        # Get linear background fit
        left_start = fit.find_nearest(bin_centers, -be_e*sigma)
        left_end = fit.find_nearest(bin_centers, -be_s*sigma)
        right_start = fit.find_nearest(bin_centers, be_s*sigma)
        right_end = fit.find_nearest(bin_centers, be_e*sigma)
        background_hist = np.concatenate((hist[left_start:left_end],
                                          hist[right_start:right_end]))
        background_bins = np.concatenate((bin_centers[left_start:left_end],
                                          bin_centers[right_start:right_end]))

        # Plot linear fit
        k, m, ferr = fit.fit_linear(background_hist, background_bins)
        print('Linear (y = k*x + m)')
        print('k  = %f ± %f' % (k, ferr[0]))
        print('m  = %f ± %f' % (m, ferr[1]))
        plt.plot(xx, fit.Linear(xx, k, m), color='purple', zorder=5,
                 label='Linear\nbackground fit')
        plt.legend(loc=2)

        # Define edges
        left_idx_peak = fit.find_nearest(bin_centers, -5*sigma)
        right_idx_peak = fit.find_nearest(bin_centers, 5*sigma)
        left_idx_shoulder = fit.find_nearest(bin_centers, 3*sigma)
        right_idx_shoulder = fit.find_nearest(bin_centers, 5*sigma)

        # Get background counts
        counts_back_peak = sum(fit.Linear(bin_centers[left_idx_peak:right_idx_peak], k, m))
        counts_back_shoulder = sum(fit.Linear(bin_centers[left_idx_shoulder:right_idx_shoulder], k, m))

        # Get total counts
        counts_signal_and_back_peak = len(delta_E[(delta_E > -5*sigma) & (delta_E < 5*sigma)])
        counts_signal_and_back_shoulder = len(delta_E[(delta_E > 3*sigma) & (delta_E < 5*sigma)])

        # Get signal
        counts_signal_peak = counts_signal_and_back_peak - counts_back_peak
        counts_signal_shoulder = counts_signal_and_back_shoulder - counts_back_shoulder

        # Store shoulder areas, peaks areas, foms and incident energies
        fom_value = counts_signal_shoulder/counts_signal_peak
        peak_counts = counts_signal_peak
        shoulder_counts = counts_signal_shoulder
        incident_energy = Ei_in_meV

    # Cross-checking
    print()
    print('--------------------------------------------------')
    print('STATISTICS')
    print('--------------------------------------------------')
    print('Shoulder')
    print('Total: %.2f counts' % counts_signal_and_back_shoulder)
    print('Background: %.2f counts' % counts_back_shoulder)
    print('Signal: %.2f counts' % counts_signal_shoulder)
    print()
    print('Peak')
    print('Total: %.2f counts' % counts_signal_and_back_peak)
    print('Background: %.2f counts' % counts_back_peak)
    print('Signal: %.2f counts' % counts_signal_peak)
    print('---')
    
    return fom_value, fom_value_min, fom_value_max