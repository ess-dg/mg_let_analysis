#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mg_common_plot.py: Contains common functions for plotting data
"""
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd

import scipy.signal as sci_sig
import calculations.energy as e_calc
import calculations.fitting as fit
import calculations.distance_calibration as dc

import plotly as py
import plotly.graph_objs as go

# ==============================================================================
#                               TOF HISTOGRAM
# ==============================================================================

def tof_histogram_plot(tof_in_us, number_bins, label=None, color=None, run='',
                       linestyle='solid', weights=None):
    """
    Function to plot a time-of-flight histogram.

    Args:
        tof_in_us (np.array): Time-of-flight in us
        number_bins (int): Number of bins in the histogram
        label (str): Label on data

    Yields:
        A histogram of time-of-flight

    Returns:
        hist (np.array): tof histogram
        bins (np.array): bin edges

    """

    # Plot
    hist, bins, __ = plt.hist(tof_in_us, range=[0, 100000], bins=number_bins,
                              histtype='step', zorder=10,
                              label=label, color=color, linestyle=linestyle,
                              weights=weights
                              )
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.grid(True, which='major', linestyle='--', zorder=0)
    #plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.yscale('log')
    #plt.title('Time-of-flight histogram')
    plt.xlabel('tof ($\mu$s)')
    plt.ylabel('counts')
    plt.title('Time-of-flight histogram')
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
                          label=None, color=None, run='', linestyle='solid',
                          weights=None):
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
                              label=label, color=color, linestyle=linestyle,
                              weights=weights
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
                         moderator_to_sample_in_m, number_bins, fig, Ei_in_meV_for_region,
                         label=None, color=None, run='', normalize_by_counts=False,
                         linestyle='-', weights=None, zorder=2, factor=1,
                         normalize_by_max=False, marker='.', markerfacecolor=None, include_hist=False):
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
    delta_E_limit = Ei_in_meV_for_region * factor # 0.2
    dE_area = delta_E[(delta_E >= -delta_E_limit) & (delta_E <= delta_E_limit)].shape[0]
    if normalize_by_counts:
        hist, bins, __ = plt.hist(delta_E, range=[-delta_E_limit, delta_E_limit],
                                  bins=number_bins, histtype='step', 
                                  label=label, color=color, weights=(1/dE_area)*np.ones(len(delta_E)),
                                  linestyle=linestyle, zorder=zorder
                                  )
        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.errorbar(bin_centers, hist, np.sqrt(hist/(1/dE_area))*(1/dE_area), color=color, label=None, linestyle='')
    elif normalize_by_max:
        hist, bins = np.histogram(delta_E, range=[-delta_E_limit, delta_E_limit], bins=number_bins)
        #print('Delta E', bins[1] - bins[0])
        idxs_one = np.where(hist == 1)
        norm = 1/max(hist)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        if include_hist:
            plt.hist(delta_E, range=[-delta_E_limit, delta_E_limit], bins=number_bins, weights=norm*np.ones(len(delta_E)),
                     color=color, histtype='step', linestyle=linestyle)
        hist, bins = np.histogram(delta_E, range=[-delta_E_limit, delta_E_limit], bins=number_bins, weights=norm*np.ones(len(delta_E)))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        # Extract error bars
        errors_up = np.sqrt(hist/(norm))*(norm)
        errors_down = np.sqrt(hist/(norm))*(norm)
        errors_down[idxs_one] = 0
        plt.errorbar(bin_centers, hist, (errors_down, errors_up), color=color, label=label, linestyle='',
                     marker=marker, markerfacecolor=markerfacecolor, capsize=3)
        
    else:
        hist, bins, __ = plt.hist(delta_E, range=[-delta_E_limit, delta_E_limit],
                                  bins=number_bins, histtype='step', 
                                  label=label, color=color, linestyle=linestyle, weights=weights, zorder=zorder
                                  )
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.yscale('log')
    plt.title('Energy transfer histogram, E$_i$=%.2f meV (%.2f ??)' % (Ei_in_meV,
                                                                      e_calc.meV_to_A(Ei_in_meV)))
    plt.xlabel('E$_i$ - E$_f$ (meV)')
    plt.ylabel('counts')
    plt.xlim(-delta_E_limit, delta_E_limit)
    hist_no_zeros = hist[hist > 0]
    if len(hist_no_zeros) > 1:
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
    return hist, bins, dE_area


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
        print('Gaussian (y=a*exp(-(x-x0)^2/(2*??^2)))')
        print('a  = %f ?? %f' % (a, perr[0]))
        print('x0 = %f ?? %f' % (x0, perr[1]))
        print('??  = %f ?? %f' % (sigma, perr[2]))
        print()

        # Plot Gaussian peak fit
        min_val, max_val = - 0.2 * Ei_in_meV_high_res, 0.2 * Ei_in_meV_high_res
        xx = np.linspace(min_val, max_val, 1000)
        plt.plot(xx, fit.Gaussian(xx, a, x0, sigma), color='red', zorder=5,
                 label='Gaussian fit')

        # Plot deliminaters
        plt.vlines(3*sigma, 1e-3, 1e8, color='blue', label='3??')
        plt.vlines(5*sigma, 1e-3, 1e8, color='green', label='??5??')
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
        print('k  = %f ?? %f' % (k, ferr[0]))
        print('m  = %f ?? %f' % (m, ferr[1]))
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

# ==============================================================================
#                        PLOT HELIUM-3 AND MULTI-GRID
# ==============================================================================

def plot_mg_and_he3_3d(df_mg, df_he3, he3_mapping, mg_offset, mg_theta, region_edges=None):
    # Helium-3 tubes
    hist, __ = np.histogram(df_he3['pixel_id'], bins=len(he3_mapping['x']),
                            range=[0, len(he3_mapping['x'])])
    hist_non_zero_indices = np.where(hist != 0)[0]
    # Define labels
    labels = []
    for i in np.arange(0, len(he3_mapping['x']), 1):
        x, y, z = he3_mapping['x'][i], he3_mapping['y'][i], he3_mapping['z'][i]
        angle_zx = np.arctan(abs(x)/abs(z)) * (180/np.pi)
        angle_y_to_zx = np.arctan(abs(y)/np.sqrt(x**2 + z**2)) * (180/np.pi)
        labels.append('ID: %d<br>Counts: %d<br>??: %.2f??<br>??: %.2f??<br>arctan(abs(x)/abs(z)): %.2f??<br>arctan(abs(y)/sqrt(x^2+z^2): %.2f??' % (i, hist[i],
                                                                                                        he3_mapping['theta'][i],
                                                                                                        he3_mapping['phi'][i],
                                                                                                        angle_zx,
                                                                                                        angle_y_to_zx))
    labels = np.array(labels)
    # Plot
    trace_1 = go.Scatter3d(
        x=he3_mapping['x'][hist_non_zero_indices],
        y=he3_mapping['y'][hist_non_zero_indices],
        z=he3_mapping['z'][hist_non_zero_indices],
        mode='markers',
        marker=dict(
            size=4,
            color=np.log10(hist[hist_non_zero_indices]),
            colorscale='Jet',
            opacity=1,
            colorbar=dict(thickness=10,
                          title='log10(counts)'
                          ),
        ),
        name='Helium-3 tubes',
        text=labels[hist_non_zero_indices]
    )
    
    # Sample
    trace_2 = go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        marker=dict(
            size=4,
            line=dict(
                color='rgba(0, 0, 0, 1)',
                width=5.0
            ),
            opacity=1.0
        ),
        name='Sample'
    )
    
    # Multi-Grid vessel
    b_traces = create_mg_vessel(mg_offset, mg_theta)
    
    # Multi-Grid
    H, __ = np.histogramdd(df_mg[['wch', 'gch']].values,
                           bins=(96, 37),
                           range=((0, 96), (96, 133)))
    # Insert results into an array
    hist = [[], [], [], []]
    loc = 0
    labels = []
    for wch in range(0, 96):
        for gch in range(96, 133):
            x, y, z = dc.get_global_xyz(wch, gch, mg_offset, mg_theta)
            angle_zx = np.arctan(abs(x)/abs(z)) * (180/np.pi)
            hist[0].append(x)
            hist[1].append(y)
            hist[2].append(z)
            hist[3].append(H[wch, gch-96])
            loc += 1
            labels.append('Wire channel: ' + str(wch) + '<br>'
                          + 'Grid channel: ' + str(gch) + '<br>'
                          + 'Counts: ' + str(H[wch, gch-96]) + '<br>'
                          + '2$\theta$: %.2f'  % angle_zx
                          )
        
    # Produce 3D histogram plot
    MG_3D_trace = go.Scatter3d(x=hist[0],
                               y=hist[1],
                               z=hist[2],
                               mode='markers',
                               marker=dict(size=5,
                                           color=hist[3],
                                           colorscale='Jet',
                                           opacity=1,
                                           #colorbar=dict(thickness=20,
                                           #              title='Multi-Grid<br>(counts)'
                                           #              ),
                                           ),
                               text=labels,
                               name='Multi-Grid',
                               scene='scene1'
                               )
    trace_1.marker.colorbar.x = 0.05
    MG_3D_trace.marker.colorbar.x = 0.15
    data = [trace_1, trace_2, MG_3D_trace]
    for b_trace in b_traces:
        data.append(b_trace)

    # Include region of interest in Helium-3 tubes
    if region_edges is not None:
        trace_4 = go.Scatter3d(
                    x=he3_mapping['x'][region_edges]-0.05,
                    y=he3_mapping['y'][region_edges],
                    z=he3_mapping['z'][region_edges]-0.05,
                    mode='lines',
                    line=dict(color='rgba(0, 0, 0, 1)',
                              width=5),
                    name='Region of interest',
                    )
        data.append(trace_4)

    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-1.5, y=1.5, z=-1.5)
    )
    
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        scene=dict(
                camera=camera,
                xaxis = dict(
                        title='x (m)'),
                yaxis = dict(
                        title='y (m)'),
                zaxis = dict(
                        title='z (m)'),
                aspectmode='data')
    )
    
    fig = go.Figure(data=data, layout=layout)
    fig.layout.showlegend = False
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    py.offline.plot(fig,
                    filename='../output/mg_and_he3_hist.html',
                    auto_open=True)

    
def plot_mg_and_he3_hist_3d(df_mg, df_he3, he3_mapping, mg_offset, mg_theta,
                            region_edges=None, cmin=0, cmax=5000):
    # Helium-3 tubes
    weights = df_he3.iloc[:, :].sum(axis=1).to_numpy()
    indices = df_he3.index.values
    number_bins = len(df_he3.index.values) - 1
    hist, bins = np.histogram(indices, bins=len(df_he3.index.values), weights=weights)
    hist_non_zero_indices = np.where(hist != 0)[0]
    labels = []
    for weight, index in zip(weights, indices):
        labels.append('Counts: %d<br>ID %d' % (weight, index))
    labels = np.array(labels)

    # Plot
    trace_1 = go.Scatter3d(
        x=he3_mapping['x'][hist_non_zero_indices],
        y=he3_mapping['y'][hist_non_zero_indices],
        z=he3_mapping['z'][hist_non_zero_indices],
        mode='markers',
        text=labels[hist_non_zero_indices],
        marker=dict(
            size=4,
            color=np.log10(hist[hist_non_zero_indices]),
            colorscale='Jet',
            opacity=1,
            cmin = np.log10(cmin),
            cmax = np.log10(cmax),
            colorbar=dict(thickness=10,
                          title='Helium-3 tubes<br>(log10(counts))'
                          ),
        ),
        name='Helium-3 tubes'
    )
    
    # Sample
    trace_2 = go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        marker=dict(
            size=4,
            line=dict(
                color='rgba(0, 0, 0, 1)',
                width=5.0
            ),
            opacity=1.0
        ),
        name='Sample'
    )
    
    # Multi-Grid vessel
    b_traces = create_mg_vessel(mg_offset, mg_theta)
    
    # Multi-Grid
    H, __ = np.histogramdd(df_mg[['wch', 'gch']].values,
                           bins=(96, 37),
                           range=((0, 96), (96, 133)))
    # Insert results into an array
    hist = [[], [], [], []]
    loc = 0
    labels = []
    for wch in range(0, 96):
        for gch in range(96, 133):
            x, y, z = dc.get_global_xyz(wch, gch, mg_offset, mg_theta)
            angle_zx = np.arctan(abs(x)/abs(z)) * (180/np.pi)
            hist[0].append(x)
            hist[1].append(y)
            hist[2].append(z)
            hist[3].append(H[wch, gch-96])
            loc += 1
            labels.append('Wire channel: ' + str(wch) + '<br>'
                          + 'Grid channel: ' + str(gch) + '<br>'
                          + 'Counts: ' + str(H[wch, gch-96]) + '<br>'
                          + 'zx-angle: %.2f'  % angle_zx
                          )
    
    #hist_mg_non_zero_indices = np.where(hist[3] != 0)[0]
    
    # Produce 3D histogram plot
    MG_3D_trace = go.Scatter3d(x=hist[0],
                               y=hist[1],
                               z=hist[2],
                               mode='markers',
                               marker=dict(size=5,
                                           color=np.log10(hist[3]),
                                           colorscale='Jet',
                                           cmin=np.log10(cmin),
                                           cmax=np.log10(cmax),
                                           opacity=1,
                                           colorbar=dict(thickness=20,
                                                         title='Multi-Grid<br>(log10(counts))'
                                                         ),
                                           ),
                               text=labels,
                               name='Multi-Grid',
                               scene='scene1'
                               )
    trace_1.marker.colorbar.x = 0.05
    MG_3D_trace.marker.colorbar.x = 0.15
    data = [trace_1, trace_2, MG_3D_trace]
    for b_trace in b_traces:
        data.append(b_trace)

    # Include region of interest in Helium-3 tubes
    if region_edges is not None:
        trace_4 = go.Scatter3d(
                    x=he3_mapping['x'][region_edges],
                    y=he3_mapping['y'][region_edges],
                    z=he3_mapping['z'][region_edges],
                    mode='lines',
                    line=dict(color='rgba(0, 0, 0, 1)',
                              width=15),
                    name='Region of interest',
                    )
        data.append(trace_4)

    camera = dict(
        up=dict(x=0, y=1, z=0),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-1.5, y=1.5, z=-1.5)
    )
    
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        scene=dict(
                camera=camera,
                xaxis = dict(
                        title='x (m)'),
                yaxis = dict(
                        title='y (m)'),
                zaxis = dict(
                        title='z (m)'),
                aspectmode='data')
    )
    
    fig = go.Figure(data=data, layout=layout)
    fig.layout.showlegend = False
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    py.offline.plot(fig,
                    filename='../output/mg_and_he3_hist.html',
                    auto_open=True)
    
def create_mg_vessel(offset, theta):
    # Initiate border lines for vessel
    vessel_top = np.array([np.array([-sys.float_info.epsilon, -sys.float_info.epsilon, -0.310, -0.310, sys.float_info.epsilon]),
                           np.array([0, 0.310, 0.310, sys.float_info.epsilon, 0]),
                           np.array([0, 0, 0, 0, 0])])
    vessel_back = np.array([np.array([-sys.float_info.epsilon, -sys.float_info.epsilon, -0.310, -0.310]),
                            np.array([0, 0, sys.float_info.epsilon, sys.float_info.epsilon]),
                            np.array([0, -1.054, -1.054, 0])])
    vessel_front = np.array([np.array([-sys.float_info.epsilon, -sys.float_info.epsilon, -0.310, -0.310]),
                             np.array([0.310, 0.310, 0.310, 0.310]),
                             np.array([0, -1.054, -1.054, 0])])
    vessel_bottom = np.array([np.array([-sys.float_info.epsilon, -sys.float_info.epsilon, -0.310, -0.310, -sys.float_info.epsilon]),
                              np.array([0, 0.310, 0.310, sys.float_info.epsilon, 0]),
                              np.array([-1.054, -1.054, -1.054, -1.054, -1.054])])
    # Initiate border lines for bottom
    bottom_front = np.array([np.array([-0.362, -0.362, 0.052, 0.052, -0.362]),
                             np.array([0.358, 0.358, 0.358, 0.358, 0.358]),
                             np.array([-1.054, -1.089, -1.089, -1.054, -1.054])])
    
    bottom_back = np.array([np.array([-0.362, -0.362, 0.052, 0.052, -0.362]),
                            np.array([-0.256, -0.256, -0.256, -0.256, -0.256]),
                            np.array([-1.054, -1.089, -1.089, -1.054, -1.054])])
    
    bottom_left = np.array([np.array([-0.362, -0.362, -0.362, -0.362, -0.362]),
                            np.array([0.358, 0.358, -0.256, -0.256, 0.358]),
                            np.array([-1.054, -1.089, -1.089, -1.054, -1.054])])
    
    bottom_right = np.array([np.array([0.052, 0.052, 0.052, 0.052, 0.052]),
                             np.array([0.358, 0.358, -0.256, -0.256, 0.358]),
                             np.array([-1.054, -1.089, -1.089, -1.054, -1.054])])
    
    lines = [vessel_top, vessel_front, vessel_back, vessel_bottom, bottom_front, bottom_left, bottom_back, bottom_right]
    
    b_traces = []
    for j, line in enumerate(lines):
        xs = np.empty([len(line[0])], dtype='float')
        ys = np.empty([len(line[0])], dtype='float')
        zs = np.empty([len(line[0])], dtype='float')
        
        for i, (x, y, z) in enumerate(zip(line[1], line[2], line[0])):
            xs[i] = dc.get_new_x(x, z, theta) + offset['x']
            ys[i] = y + offset['y']
            zs[i] = dc.get_new_z(x, z, theta) + offset['z']
            
        b_trace = go.Scatter3d(x=xs,
                               y=ys,
                               z=zs,
                               mode='lines',
                               line = dict(
                                            color='rgba(0, 0, 0, 0.5)',
                                            width=5)
                                )
        b_traces.append(b_trace)
    return b_traces