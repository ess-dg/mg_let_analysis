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

import plotting.helper_functions as plotting_hf
import file_handling.mg_manage as mg_manage

# ==============================================================================
#                                   PHS (1D)
# ==============================================================================

def phs_1d_plot(clusters, clusters_uf, number_bins, bus, duration):
    """
    Plot the 1D PHS with and without filters.

    Args:
        clusters (DataFrame): Clustered events, filtered
        clusters_uf (DataFrame): Clustered events, unfiltered
        number_bins (int): Number of bins to use in the histogram
        bus (int): Bus to plot
        duration (float): Measurement duration in seconds

    Yields:
        Figure containing a 1D PHS plot
    """
    # Clusters filtered
    plt.hist(clusters.wadc, bins=number_bins, histtype='step',
             zorder=5, range=[0, 8000], label='Wires (filtered)', color='blue',
             weights=(1/duration)*np.ones(len(clusters.wadc)))
    plt.hist(clusters.gadc, bins=number_bins, histtype='step',
             zorder=5, range=[0, 8000], label='Grids (filtered)', color='red',
             weights=(1/duration)*np.ones(len(clusters.gadc)))
    # Clusters unfiltered
    hist_w, bins_w, __ = plt.hist(clusters_uf.wadc, bins=number_bins,
                                  histtype='step', zorder=5, range=[0, 8000],
                                  label='Wires', color='cyan',
                                  weights=(1/duration)*np.ones(len(clusters_uf.wadc)))
    hist_g, bins_g, __ = plt.hist(clusters_uf.gadc, bins=number_bins,
                                  histtype='step', zorder=5, range=[0, 8000],
                                  label='Grids', color='magenta',
                                  weights=(1/duration)*np.ones(len(clusters_uf.gadc)))
    plt.title('Bus: %d' % bus)
    plt.xlabel('Charge (ADC channels)')
    plt.ylabel('Counts/s')
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    #plt.ylim(1e-5, 1)
    plt.legend()
    # Save histograms
    bins_w_c = 0.5 * (bins_w[1:] + bins_w[:-1])
    bins_g_c = 0.5 * (bins_g[1:] + bins_g[:-1])
    dirname = os.path.dirname(__file__)
    path_wires = os.path.join(dirname,
                              '../output/seq_phs_unfiltered_wires_bus_%d.txt' % bus)
    path_grids = os.path.join(dirname,
                              '../output/seq_phs_unfiltered_grids_bus_%d.txt' % bus)
    np.savetxt(path_wires,
               np.transpose(np.array([bins_w_c, hist_w])),
               delimiter=",",header='bins, hist (counts/s)')
    np.savetxt(path_grids,
               np.transpose(np.array([bins_g_c, hist_g])),
               delimiter=",",header='bins, hist (counts/s)')


# ==============================================================================
#                                   PHS (2D)
# ==============================================================================

def phs_2d_plot(events, bus, vmin, vmax):
    """
    Histograms the ADC-values from each channel individually and summarises it
    in a 2D histogram plot, where the color scale indicates number of counts.
    Each bus is presented in an individual plot.

    Args:
        events (DataFrame): DataFrame containing individual events
        bus (int): Bus to plot
        vmin (float): Minimum value on color scale
        vmax (float): Maximum value on color scale

    Yields:
        Plot containing 2D PHS heatmap plot
    """
    plt.xlabel('Channel')
    plt.ylabel('Charge (ADC channels)')
    plt.title('Bus: %d' % bus)
    bins = [120, 120]
    if events.shape[0] > 1:
        plt.hist2d(events.ch, events.adc, bins=bins,
                   norm=LogNorm(vmin=vmin, vmax=vmax),
                   range=[[-0.5, 119.5], [0, 4400]],
                   cmap='jet')
    cbar = plt.colorbar()
    cbar.set_label('Counts')


# ==============================================================================
#                           PHS (Wires vs Grids)
# ==============================================================================


def clusters_phs_plot(clusters, bus, duration, vmin, vmax):
    """
    Histograms ADC charge from wires vs grids, one for each bus, showing the
    relationship between charge collected by wires and charge collected by
    grids. In the ideal case, there should be linear relationship between these
    two quantities.

    Args:
        clusters (DataFrame): Clustered events
        bus (int): Bus to plot
        duration (float): Measurement duration in seconds
        vmin (float): Minimum value on color scale
        vmax (float): Maximum value on color scale

    Yields:
        Plot containing the PHS 2D scatter plot

    """

    plt.xlabel('Charge wires (ADC channels)')
    plt.ylabel('Charge grids (ADC channels)')
    plt.title('Bus: %d' % bus)
    bins = [200, 200]
    ADC_range = [[0, 10000], [0, 10000]]
    plt.hist2d(clusters.wadc, clusters.gadc, bins=bins,
               norm=LogNorm(vmin=vmin, vmax=vmax),
               range=ADC_range,
               cmap='jet',
               weights=(1/duration)*np.ones(len(clusters.wadc)))
    cbar = plt.colorbar()
    cbar.set_label('Counts/s')


# ==============================================================================
#                          COINCIDENCE HISTOGRAM (2D)
# ==============================================================================

def clusters_2d_plot(clusters, title, vmin, vmax, duration):
    """
    Plots a 2D histograms of clusters: wires vs grids.

    Args:
        clusters (DataFrame): Clustered events, filtered
        vmin (float): Minimum value on color scale
        vmax (float): Maximum value on color scale
        duration (float): Measurement duration in seconds

    Yields:
        Plot containing the 2D coincidences

    """

    plt.hist2d(clusters.wch, clusters.gch, bins=[80, 40],
               range=[[-0.5, 79.5], [79.5, 119.5]],
               norm=LogNorm(vmin=vmin, vmax=vmax),
               cmap='jet',
               weights=(1/duration)*np.ones(len(clusters.wch)))
    plt.xlabel('Wire (Channel number)')
    plt.ylabel('Grid (Channel number)')
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label('Counts/s')

# ==============================================================================
#                               MULTIPLICITY
# ==============================================================================

def multiplicity_plot(clusters, bus, duration, vmin=None, vmax=None):
    """
    Plots a 2D histograms of wire-grid event multiplicity in the clusters

    Args:
        clusters (DataFrame): Clustered events, filtered
        bus (int): Bus to plot
        duration (float): Measurement duration in seconds
        vmin (float): Minimum value on color scale
        vmax (float): Maximum value on color scale

    Yields:
        Plot containing the 2D multiplicity distribution

    """
    # Declare parameters
    m_range = [0, 10, 0, 10]
    # Plot data
    hist, xbins, ybins, im = plt.hist2d(clusters.wm, clusters.gm,
                                        bins=[m_range[1]-m_range[0]+1,
                                              m_range[3]-m_range[2]+1],
                                        range=[[m_range[0], m_range[1]+1],
                                               [m_range[2], m_range[3]+1]],
                                        norm=LogNorm(vmin=vmin, vmax=vmax),
                                        cmap='jet',
                                        weights=(1/duration)*np.ones(len(clusters.wm)))
    # Iterate through all squares and write percentages
    tot = clusters.shape[0] * (1/duration)
    font_size = 6.5
    for i in range(len(ybins)-1):
        for j in range(len(xbins)-1):
            if hist[j, i] > 0:
                text = plt.text(xbins[j]+0.5, ybins[i]+0.5,
                                '%.0f%%' % (100*(hist[j, i]/tot)),
                                color="w", ha="center", va="center",
                                fontweight="bold", fontsize=font_size)
                text.set_path_effects([path_effects.Stroke(linewidth=1,
                                                           foreground='black'),
                                       path_effects.Normal()])
    # Set ticks on axis
    ticks_x = np.arange(m_range[0], m_range[1]+1, 1)
    locs_x = np.arange(m_range[0] + 0.5, m_range[1]+1.5, 1)
    ticks_y = np.arange(m_range[2], m_range[3]+1, 1)
    locs_y = np.arange(m_range[2] + 0.5, m_range[3]+1.5, 1)
    plt.xticks(locs_x, ticks_x)
    plt.yticks(locs_y, ticks_y)
    plt.xlabel("Wire multiplicity")
    plt.ylabel("Grid multiplicity")
    cbar = plt.colorbar()
    cbar.set_label('Counts/s')
    plt.title('Bus: %d' % bus)
    #plt.tight_layout()


# ==============================================================================
#                                       RATE
# ==============================================================================


def rate_plot(clusters, number_bins, bus):
    """
    Histograms the rate as a function of time.

    Args:
        clusters (DataFrame): Clustered events
        number_bins (int): The number of bins to histogram the data into
        bus (int): Bus to plot

    Yields:
        Plot containing the rate as a function of time

    """

    # Prepare figure
    plt.title('Bus: %d' % bus)
    plt.xlabel('Time (hours)')
    plt.ylabel('Rate (events/s)')
    plt.grid(True, which='major', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    # Plot
    time = (clusters.time * 62.5e-9)/(60 ** 2)
    hist, bin_edges = np.histogram(time, bins=number_bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    delta_t = (60 ** 2) * (bin_centers[1] - bin_centers[0])
    plt.errorbar(bin_centers, (hist/delta_t), np.sqrt(hist)/delta_t,
                 marker='.', linestyle='', zorder=5, color='black')


# ==============================================================================
#                              UNIFORMITY - GRIDS
# ==============================================================================

def grid_histogram(clusters, bus, duration):
    """
    Histograms the counts in each grid.

    Args:
        clusters (DataFrame): Clustered events
        bus(int): The bus of the data
        duration (float): Measurement duration in seconds

    Yields:
        Plot containing the grid histogram
    """

    # Prepare figure
    plt.title('Bus: %d' % bus)
    plt.xlabel('Grid channel')
    plt.ylabel('Counts/s')
    #plt.yscale('log')
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    # Histogram data
    plt.hist(clusters.gch, bins=40, zorder=4, range=[79.5, 119.5],
             weights=(1/duration)*np.ones(len(clusters.gch)),
             histtype='step', color='black')


# ==============================================================================
#                              UNIFORMITY - WIRES
# ==============================================================================

def wire_histogram(clusters, bus, duration):
    """
    Histograms the counts in each wire.

    Args:
        clusters (DataFrame): Clustered events
        bus(int): The bus of the data
        duration (float): Measurement duration in seconds

    Yields:
        Plot containing the wire histogram
    """

    # Prepare figure
    plt.title('Bus: %d' % bus)
    plt.xlabel('Wire channel')
    plt.ylabel('Counts/s')
    #plt.yscale('log')
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    # Histogram data
    plt.hist(clusters.wch, bins=80, zorder=4, range=[-0.5, 79.5],
             weights=(1/duration)*np.ones(len(clusters.wch)),
             histtype='step', color='black')


# ==============================================================================
#                      PLOT ALL BASIC PLOTS FOR ONE BUS
# ==============================================================================

def mg_plot_basic_bus(run, bus, clusters_unfiltered, events, df_filter, area,
                      plot_title=''):
    """
    Function to plot all basic plots for SEQUOIA detector, for a single bus,
    such as PHS, Coincidences and rate.

    Ordering of plotting is:

    PHS 2D             - NOT FILTERED
    PHS 1D             - FILTERED AND NOT FILTERED
    MULTIPLICITY       - FILTERED
    PHS CORRELATION    - FILTERED
    COINCIDENCES 2D    - FILTERED
    RATE               - FILTERED
    UNIFORMITY (WIRES) - FILTERED
    UNIFORMITY (GRIDS) - FILTERED

    Note that all plots are filtered except the PHS 2D.

    Args:
        run (str): File run
        bus (int): Bus to plot
        clusters_unfiltered (DataFrame): Unfiltered clusteres
        events (DataFrame): Individual events
        df_filter (dict): Dictionary specifying the filter which will be used
                          on the clustered data
        area (float): Area in m^2 of the active detector surface
        plot_title (str): Title of PLOT

    Yields:
        Plots the basic analysis

    """

    plotting_hf.set_thick_labels(15)

    # Filter clusters
    clusters = mg_manage.filter_data(clusters_unfiltered, df_filter)

    # Declare parameters
    duration_unf = ((clusters_unfiltered.time.values[-1]
                    - clusters_unfiltered.time.values[0]) * 62.5e-9)
    duration = (clusters.time.values[-1] - clusters.time.values[0]) * 62.5e-9

    # Filter data from only one bus
    events_bus = events[events.bus == bus]
    clusters_bus = clusters[clusters.bus == bus]
    clusters_uf_bus = clusters_unfiltered[clusters_unfiltered.bus == bus]

    fig = plt.figure()
    plt.suptitle(plot_title, fontsize=15, fontweight='bold', y=1.00005)
    # PHS - 2D
    plt.subplot(4, 2, 1)
    vmin = 1
    vmax = events.shape[0] // 1000 + 100
    if events_bus.shape[0] > 0:
        phs_2d_plot(events_bus, bus, vmin, vmax)
    plt.title('PHS vs Channel')

    # PHS - 1D
    plt.subplot(4, 2, 2)
    bins_phs_1d = 300
    phs_1d_plot(clusters_bus, clusters_uf_bus, bins_phs_1d, bus, duration)
    plt.yscale('log')
    plt.title('PHS')

    # Coincidences - 2D
    plt.subplot(4, 2, 5)
    if clusters.shape[0] != 0:
        vmin = (1 * 1/duration)
        vmax = (clusters.shape[0] // 450 + 5) * 1/duration
    else:
        duration = 1
        vmin = 1
        vmax = 1

    number_events = clusters_bus.shape[0]
    number_events_error = np.sqrt(clusters_bus.shape[0])
    events_per_s = number_events/duration
    events_per_s_m2 = events_per_s/area
    events_per_s_m2_error = number_events_error/(duration*area)
    title = ('Coincidences\n(%d events, %.3f±%.3f events/s/m$^2$)' % (number_events,
                                                                      events_per_s_m2,
                                                                      events_per_s_m2_error))
    if number_events > 1:
        clusters_2d_plot(clusters_bus, title, vmin, vmax, duration)

    # Rate
    plt.subplot(4, 2, 6)
    number_bins = 40
    rate_plot(clusters_bus, number_bins, bus)
    plt.title('Rate vs time')

    # Multiplicity
    plt.subplot(4, 2, 3)
    if clusters_bus.shape[0] > 1:
        multiplicity_plot(clusters_bus, bus, duration)
    plt.title('Event multiplicity')

    # Coincidences - PHS
    plt.subplot(4, 2, 4)
    if clusters.shape[0] != 0:
        vmin = 1/duration
        vmax = (clusters.shape[0] // 450 + 1000) / duration
    else:
        duration = 1
        vmin = 1
        vmax = 1
    if clusters_bus.shape[0] > 1:
        clusters_phs_plot(clusters_bus, bus, duration, vmin, vmax)
    plt.title('Charge coincidences')

    # Uniformity - grids
    plt.subplot(4, 2, 8)
    grid_histogram(clusters_bus, bus, duration)
    plt.title('Uniformity - grids')

    # Uniformity - wires
    plt.subplot(4, 2, 7)
    wire_histogram(clusters_bus, bus, duration)
    plt.title('Uniformity - wires')

    # Save data
    fig.set_figwidth(10)
    fig.set_figheight(16)
    plt.tight_layout()
    output_path = 'output/%s_summary_bus_%d.png' % (run, bus)
    fig.savefig(output_path, bbox_inches='tight')
