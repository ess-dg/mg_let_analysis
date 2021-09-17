#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mg_basic_plot.py: Contains the basic functions to plot Multi-Grid data.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd

import scipy.signal as sci_sig
import calculations.energy as e_calc
import calculations.fitting as fit

# ==============================================================================
#                               TOF HISTOGRAM
# ==============================================================================

def tof_histogram_plot(df, number_bins, label=None, color=None, run=''):
    """
    Function to plot a time-of-flight histogram.

    Args:
        df (pd.DataFrame): DataFrame containing the data
        number_bins (int): Number of bins in the histogram
        label (str): Label on data

    Yields:
        A histogram of time-of-flight

    Returns:
        hist (np.array): tof histogram
        bins (np.array): bin edges

    """

    # Get tof
    tof_in_us = df['tof']
    # Plot
    hist, bins, __ = plt.hist(tof_in_us, range=[0, 100000], bins=number_bins,
                              histtype='step', zorder=10,
                              label=label, color=color
                              )
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.yscale('log')
    plt.title('Time-of-flight histogram')
    plt.xlabel('tof (μs)')
    plt.ylabel('counts')
    # Save data
    dirname = os.path.dirname(__file__)
    output_path = os.path.join(dirname, '../output/%s_tof_histogram.txt' % run)
    # Save as text_file
    np.savetxt(output_path,
               np.transpose(np.array([bin_centers, hist, np.sqrt(hist)])),
               delimiter=",",header='tof (us), counts, error')
    return hist, bins


# ==============================================================================
#                              ENERGY HISTOGRAM
# ==============================================================================

def energy_histogram_plot(tof_in_us, sample_to_detection_in_m, number_bins,
                          moderator_to_sample_in_m, interval=[0, 30],
                          label=None, color=None, run=''):
    """
    Function to plot an energy histogram.

    Args:
        tof_in_us (np.array): Time-of-flight values in micro-seconds
        sample_to_detection_in_m (np.array): Sample-to-detection distance in
                                             meters
        number_bins (int): Number of bins in histogra
        label (str): Label on data

    Yields:
        A histogram of energy

    Returns:
        hist (np.array): energy histogram
        bins (np.array): bin edges

    """

    # Get energies
    energies = e_calc.get_energy(tof_in_us, sample_to_detection_in_m,
                                 moderator_to_sample_in_m)
    # Plot
    hist, bins, __ = plt.hist(energies, range=[interval[0], interval[1]],
                              bins=number_bins,
                              histtype='step', zorder=2,
                              label=label, color=color
                              )
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.yscale('log')
    plt.title('Energy histogram')
    plt.xlabel('E (meV)')
    plt.ylabel('counts')
    return hist, bins

# ==============================================================================
#                            ENERGY TRANSFER HISTOGRAM
# ==============================================================================

def energy_transfer_plot(Ei_in_meV, tof_in_us, sample_to_detection_in_m,
                         moderator_to_sample_in_m, number_bins, fig,
                         label=None, color=None, run=''):
    """
    Function to plot an energy transfer histogram.

    Args:
        Ei_in_meV (float): Initial energy in meV
        tof_in_us (np.array): Time-of-flight in micro seconds
        sample_to_detection_in_m (np.array): Sample-to-detection distance in
                                             meters
        number_bins (int): Number of bins in histogram
        label (str): Label on data

    Yields:
        A histogram of energy transfer plot

    Returns:
        hist (np.array): energy transfer histogram
        bins (np.array): bin edges

    """

    # Get energy transfer data
    delta_E = e_calc.get_energy_transfer(Ei_in_meV, tof_in_us,
                                         sample_to_detection_in_m,
                                         moderator_to_sample_in_m)
    # Plot data
    delta_E_limit = Ei_in_meV * 0.2
    hist, bins, __ = plt.hist(delta_E, range=[-delta_E_limit, delta_E_limit],
                              bins=number_bins, histtype='step', zorder=2,
                              label=label, color=color
                              )
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.yscale('log')
    plt.title('Energy transfer histogram, E$_i$=%.2f meV (%.2f Å)' % (Ei_in_meV,
                                                                      e_calc.meV_to_A(Ei_in_meV)))
    plt.xlabel('E$_i$ - E$_f$ (meV)')
    plt.ylabel('counts')
    plt.xlim(-delta_E_limit, delta_E_limit)
    hist_no_zeros = hist[hist > 0]
    plt.ylim(min(hist_no_zeros)/2, max(hist_no_zeros)*2)
    # Save data
    dirname = os.path.dirname(__file__)
    output_fig = os.path.join(dirname,
                              '../output/%s_dE_%.2f_meV.png' % (run, Ei_in_meV))
    fig.savefig(output_fig, bbox_inches='tight')
    # Save as text_file
    bin_centers = (bins[:-1] + bins[1:]) / 2
    output_txt = os.path.join(dirname,
                              '../output/%s_dE_%.2f_meV.txt' % (run, Ei_in_meV))
    np.savetxt(output_txt,
               np.transpose(np.array([bin_centers, hist, np.sqrt(hist)])),
               delimiter=",",header='dE (meV), counts, error')
    return hist, bins


# ==============================================================================
#                         GET ALL FIGURE-OF-MERITS
# ==============================================================================

def get_all_foms(tof_in_us, sample_to_detection_in_m, number_bins,
                 moderator_to_sample_in_m, prominence=1e4, run=''):
    """
    Function to get all figure-of-merits (foms) and visualize the procedure.

    Args:
        tof_in_us (np.array): Time-of-flight in micro seconds
        sample_to_detection_in_m (np.array): Sample-to-detection distance in
                                             meters
        number_bins (int): Number of bins in histogram
        prominence (float): Peak prominence for the peak search algorithm to
                            find
        run (str): Run identifier

    Returns:
        incident_energies (np.array): Array with incident energies
        fom_values (np.array): Array with foms
        peak_counts (np.array): Array with areas in the peaks
        shoulder_counts (np.array): Array with areas in shoulder

    """

    # Produce an energy histogram over all energies
    energies = e_calc.get_energy(tof_in_us, sample_to_detection_in_m,
                                 moderator_to_sample_in_m)
    hist, bin_edges = np.histogram(energies, bins=1000, range=[0, 30])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Get peaks
    peak_idxs = sci_sig.find_peaks(hist, prominence=prominence)[0]
    Eis_in_meV = bin_centers[peak_idxs]

    # Initiate vector to store figure-of-merit
    fom_values = np.empty(len(Eis_in_meV), dtype=float)
    peak_counts = np.empty(len(Eis_in_meV), dtype=float)
    shoulder_counts = np.empty(len(Eis_in_meV), dtype=float)
    incident_energies = np.empty(len(Eis_in_meV), dtype=float)

    # Plot all energy transfer plots
    for i, Ei_in_meV in enumerate(Eis_in_meV):

        # Get incident energy to a high resolution
        lower_limit = Ei_in_meV - (Ei_in_meV * 0.2)
        upper_limit = Ei_in_meV + (Ei_in_meV * 0.2)
        hist, bin_edges = np.histogram(energies, bins=number_bins,
                                       range=[lower_limit, upper_limit])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        idx_max_value = np.argmax(hist)
        Ei_in_meV_high_res = bin_centers[idx_max_value]

        # Plot data
        fig = plt.figure()
        hist, bin_edges = energy_transfer_plot(Ei_in_meV_high_res,
                                               tof_in_us,
                                               sample_to_detection_in_m,
                                               moderator_to_sample_in_m,
                                               number_bins,
                                               fig,
                                               color='black',
                                               run=run, label='Data')
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Get event-by-event information on energy transfer
        delta_E = e_calc.get_energy_transfer(Ei_in_meV_high_res, tof_in_us,
                                             sample_to_detection_in_m,
                                             moderator_to_sample_in_m)

        # Fit data
        be_s = 6
        be_e = 8
        a, x0, sigma, perr = fit.fit_data(hist, bin_centers)
        print('--------------------------------------------------')
        print('FITTING RESULTS')
        print('--------------------------------------------------')
        print('Gaussian (y=a*exp(-(x-x0)^2/(2*σ^2)))')
        print('a  = %f ± %f' % (a, perr[0]))
        print('x0 = %f ± %f' % (x0, perr[1]))
        print('σ  = %f ± %f' % (sigma, perr[2]))
        print()

        # Plot Gaussian peak fit
        min_val, max_val = - 0.2 * Ei_in_meV_high_res, 0.2 * Ei_in_meV_high_res
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
        fom_values[i] = counts_signal_shoulder/counts_signal_peak
        peak_counts[i] = counts_signal_peak
        shoulder_counts[i] = counts_signal_shoulder
        incident_energies[i] = Ei_in_meV_high_res

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

        # Save data
        file_path = '../output/%s_gaussian_fit_%.d_meV.png' % (run,
                                                               Ei_in_meV_high_res*100)
        dirname = os.path.dirname(__file__)
        output_path = os.path.join(dirname, file_path)
        fig.savefig(output_path, bbox_inches='tight')
        plt.show()
    return incident_energies, fom_values, peak_counts, shoulder_counts
