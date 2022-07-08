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
import plotly as py
import plotly.graph_objs as go

import plotting.helper_functions as plotting_hf

import file_handling.mg_seq.mg_seq_manage as mg_manage

import calculations.distance_calibration as dc
import calculations.energy as e_calc

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

def phs_2d_plot(events, vmin, vmax):
    """
    Histograms the ADC-values from each channel individually and summarises it
    in a 2D histogram plot, where the color scale indicates number of counts.
    Each bus is presented in an individual plot.

    Args:
        events (DataFrame): DataFrame containing individual events
        vmin (float): Minimum value on color scale
        vmax (float): Maximum value on color scale

    Yields:
        Plot containing 2D PHS heatmap plot
    """
    plt.xlabel('Channel')
    plt.ylabel('Charge (ADC channels)')
    bins = [133, 4095]
    if events.shape[0] > 1:
        plt.hist2d(events.ch, events.adc, bins=bins,
                   norm=LogNorm(vmin, vmax),
                   range=[[-0.5, 132.5], [0, 4400]],
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
               norm=LogNorm(),
               #vmin=vmin, vmax=vmax,
               range=ADC_range,
               cmap='jet',
               weights=(1/duration)*np.ones(len(clusters.wadc)))
    cbar = plt.colorbar()
    cbar.set_label('Counts/s')
    
# ==============================================================================
#                               TOF
# ==============================================================================


def clusters_tof_plot(clusters, file_name, interval=None, bins=100, label=None):
    """

    """
    plt.xlabel('tof (s)')
    plt.ylabel('Counts')
    plt.hist(clusters.tof * 62.5e-9, bins=bins, histtype='step', range=interval, label=label)
    plt.yscale('log')
    plt.title(file_name)
    
def clusters_tof_per_voxel(df, file_name, number_bins=100):
    """

    """
    # Create folder for histograms
    dirname = os.path.dirname(__file__)
    folder_path = os.path.join(dirname, '../output/%s_tof_histograms' % file_name)
    mkdir_p(folder_path)
    wchs = np.arange(0, 96, 1)
    gchs = np.arange(96, 133, 1)
    for wch in wchs:
        for gch in gchs:
            df_voxel = df[(df.gch == gch) & (df.wch == wch)]
            plt.xlabel('tof (s)')
            plt.ylabel('Counts')
            hist, bins = np.histogram(df_voxel.tof * 62.5e-9, bins=number_bins)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            plt.yscale('log')
            plt.title('wch=%d, gch=%d, file: %s' % (wch, gch, file_name))
            # Save data
            dirname = os.path.dirname(__file__)
            output_path = '%s/wch_%d_gch_%d.txt' % (folder_path, wch, gch)
            # Save as text_file
            np.savetxt(output_path,
                       np.transpose(np.array([bin_centers, hist, np.sqrt(hist)])),
                       delimiter=",",header='tof (us), counts, error')


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

    plt.hist2d(clusters.wch, clusters.gch, bins=[96, 37],
               range=[[-0.5, 95.5], [95.5, 132.5]],
               #norm=LogNorm(vmin=vmin, vmax=vmax),
               #vmin=vmin, vmax=vmax,
               cmap='jet',
               #weights=(1/duration)*np.ones(len(clusters.wch))
              )
    
    plt.xlabel('Wire (Channel number)')
    plt.ylabel('Grid (Channel number)')
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label('Counts')
    
# ==============================================================================
#                          COINCIDENCE HISTOGRAM (3D)
# ==============================================================================

def clusters_3d_plot(df, title):
    """
    Plots a 3D histograms of clusters.

    Args:
        df (DataFrame): Clustered events
        title (str): Title of plot

    Yields:
        Plot containing the 3D coincidences

    """
    H, __ = np.histogramdd(df[['wch', 'gch']].values,
                           bins=(96, 37),
                           range=((0, 96), (96, 133)))
    # Insert results into an array
    hist = [[], [], [], []]
    loc = 0
    labels = []
    for wch in range(0, 96):
        for gch in range(96, 133):
            x, y, z = dc.get_local_xyz(wch, gch)
            hist[0].append(x)
            hist[1].append(y)
            hist[2].append(z)
            hist[3].append(H[wch, gch-96])
            loc += 1
            labels.append('Wire channel: ' + str(wch) + '<br>'
                          + 'Grid channel: ' + str(gch) + '<br>'
                          + 'Counts: ' + str(H[wch, gch-96])
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
                                           colorbar=dict(thickness=20,
                                                         title='Counts'
                                                         ),
                                           ),
                               text=labels,
                               name='Multi-Grid',
                               scene='scene1'
                               )
    # Introduce figure and put everything together
    fig = py.subplots.make_subplots(rows=1, cols=1, specs=[[{'is_3d': True}]])
    # Insert histogram
    fig.append_trace(MG_3D_trace, 1, 1)
    fig['layout']['scene1']['xaxis'].update(title='x (m)')#, range=[0, 1.1])
    fig['layout']['scene1']['yaxis'].update(title='y (m)')#, range=[0, 1.1])
    fig['layout']['scene1']['zaxis'].update(title='z (m)')#, range=[-1.1, 0])
    fig['layout']['scene1'].update(aspectmode='data')
    fig['layout'].update(title='Coincidences (3D)<br>Data set: ' + str(title) + '.pcapng')
    fig.layout.showlegend = False
    # Plot
    py.offline.init_notebook_mode()
    #py.offline.iplot(fig)
    py.offline.plot(fig,
                    filename='../output/coincident_events_histogram.html',
                    auto_open=True)


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
                                        norm=LogNorm(),
                                        cmap='jet',
                                        weights=(1/duration)*np.ones(len(clusters.wm)))
    # Iterate through all squares and write percentages
    tot = clusters.shape[0] * (1/duration)
    font_size = 10.5
    for i in range(len(ybins)-1):
        for j in range(len(xbins)-1):
            if hist[j, i] > 0:
                text = plt.text(xbins[j]+0.5, ybins[i]+0.5,
                                '%.0f' % (100*(hist[j, i]/tot)) + '\%',
                                color="w", ha="center", va="center",
                                #fontweight="bold",
                                fontsize=font_size)
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
    plt.hist(clusters.gch, bins=37, zorder=4, range=[95.5, 132.5],
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
    plt.hist(clusters.wch, bins=96, zorder=4, range=[-0.5, 95.5],
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

    #plotting_hf.set_thick_labels(10)

    # Filter clusters
    clusters = mg_manage.filter_data(clusters_unfiltered, df_filter)

    # Declare parameters
    duration_unf = ((clusters_unfiltered.time.values[-1]
                    - clusters_unfiltered.time.values[0]) * 62.5e-9)
    #duration = duration_unf
    if len(clusters.time.values > 0):
        duration = (clusters.time.values[-1] - clusters.time.values[0]) * 62.5e-9
    else:
        duration = -1

    # Filter data from only one bus
    events_bus = events[events.bus == bus]
    clusters_bus = clusters[clusters.bus == bus]
    clusters_uf_bus = clusters_unfiltered[clusters_unfiltered.bus == bus]

    fig = plt.figure()
    #plt.suptitle(plot_title, fontsize=15, fontweight='bold', y=0.98)
    # PHS - 2D
    plt.subplot(4, 2, 1)
    vmin = 1
    vmax = events.shape[0] // 1000 + 100
    if events_bus.shape[0] > 0:
        phs_2d_plot(events_bus, vmin, vmax)
    plt.title('PHS vs Channel')

    # PHS - 1D
    plt.subplot(4, 2, 2)
    bins_phs_1d = 300
    phs_1d_plot(clusters_bus, clusters_uf_bus, bins_phs_1d, bus, duration)
    #plt.yscale('log')
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
    output_path = '../output/%s_summary_bus_%d.png' % (run, bus)
    fig.savefig(output_path, bbox_inches='tight')
    plt.show()
    
# ==============================================================================
#                      PLOT ALL BASIC PLOTS FOR ONE BUS
# ==============================================================================

def mg_plot_basic_bus_plus(run, bus, clusters_unfiltered, events, df_filter, area,
                           Ei_in_meV, fig, dE_interval=None, plot_title=''):
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

    plotting_hf.set_thick_labels(10)

    # Filter clusters
    clusters = mg_manage.filter_data(clusters_unfiltered, df_filter)

    # Declare parameters
    duration_unf = ((clusters_unfiltered.time.values[-1]
                    - clusters_unfiltered.time.values[0]) * 62.5e-9)
    #duration = duration_unf
    duration = (clusters.time.values[-1] - clusters.time.values[0]) * 62.5e-9

    # Filter data from only one bus
    events_bus = events[events.bus == bus]
    clusters_bus_no_extra_filter = clusters[clusters.bus == bus]
    clusters_uf_bus = clusters_unfiltered[clusters_unfiltered.bus == bus]
    
    clusters_bus = clusters_bus_no_extra_filter
    # Perform additional filter based on delta E
    if dE_interval is not None:
        mg_offset = {'x': -1.612, 'y': -0.862, 'z': 2.779} 
        mg_theta = 57.181 * np.pi/180
        moderator_to_sample_in_m = 25
        tof_in_us = clusters_bus_no_extra_filter.tof.values * 62.5e-9 * 1e6
        sample_to_detection_in_m = dc.get_sample_to_detection_distances(clusters_bus_no_extra_filter['wch'], clusters_bus_no_extra_filter['gch'], mg_offset, mg_theta)
        delta_E = e_calc.get_energy_transfer(Ei_in_meV, tof_in_us, sample_to_detection_in_m, moderator_to_sample_in_m)
        dE_filter = (delta_E >= dE_interval[0]) & (delta_E <= dE_interval[1])
        clusters_bus = clusters_bus[dE_filter]


    plt.suptitle(plot_title, fontsize=15, fontweight='bold', y=0.98)
    # PHS - 2D
    plt.subplot(4, 4, 1)
    vmin = 1
    vmax = events.shape[0] // 1000 + 100
    if events_bus.shape[0] > 0:
        phs_2d_plot(events_bus, vmin, vmax)
    plt.title('PHS vs Channel')

    # PHS - 1D
    plt.subplot(4, 4, 2)
    bins_phs_1d = 300
    phs_1d_plot(clusters_bus, clusters_uf_bus, bins_phs_1d, bus, duration)
    plt.yscale('log')
    plt.title('PHS')

    # Coincidences - 2D
    plt.subplot(4, 4, 9)
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
    plt.subplot(4, 4, 10)
    number_bins = 40
    rate_plot(clusters_bus, number_bins, bus)
    plt.title('Rate vs time')

    # Multiplicity
    plt.subplot(4, 4, 5)
    if clusters_bus.shape[0] > 1:
        multiplicity_plot(clusters_bus, bus, duration)
    plt.title('Event multiplicity')

    # Coincidences - PHS
    plt.subplot(4, 4, 6)
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
    plt.subplot(4, 4, 14)
    grid_histogram(clusters_bus, bus, duration)
    plt.title('Uniformity - grids')

    # Uniformity - wires
    plt.subplot(4, 4, 13)
    wire_histogram(clusters_bus, bus, duration)
    plt.title('Uniformity - wires')
    
    # Energy transfer
    bins_dE = 100
    mg_offset = {'x': -1.612, 'y': -0.862, 'z': 2.779} 
    mg_theta = 57.181 * np.pi/180
    moderator_to_sample_in_m = 25
    delta_E_limit = Ei_in_meV*0.2
    
    tof_in_us_uf = clusters_bus_no_extra_filter.tof.values * 62.5e-9 * 1e6
    sample_to_detection_in_m_uf = dc.get_sample_to_detection_distances(clusters_bus_no_extra_filter['wch'], clusters_bus_no_extra_filter['gch'], mg_offset, mg_theta)
    delta_E_uf = e_calc.get_energy_transfer(Ei_in_meV, tof_in_us_uf, sample_to_detection_in_m_uf, moderator_to_sample_in_m)
    
    tof_in_us_f = clusters_bus.tof.values * 62.5e-9 * 1e6
    sample_to_detection_in_m_f = dc.get_sample_to_detection_distances(clusters_bus['wch'], clusters_bus['gch'], mg_offset, mg_theta)
    delta_E_f = e_calc.get_energy_transfer(Ei_in_meV, tof_in_us_f, sample_to_detection_in_m_f, moderator_to_sample_in_m)
    
    plt.subplot(4, 4, 7)
    plt.hist(delta_E_uf, range=[-delta_E_limit, delta_E_limit], bins=bins_dE, histtype='step', label='Without extra filter')
    plt.hist(delta_E_f, range=[-delta_E_limit, delta_E_limit], bins=bins_dE, histtype='step', label='With extra filter')
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.yscale('log')
    plt.xlabel('E$_i$ - E$_f$ (meV)')
    plt.ylabel('counts')
    plt.legend()
    plt.title('Energy transfer at %.2f meV' % Ei_in_meV)
    
    
    # Time-of-Flight, zoom
    plt.subplot(4, 4, 4)
    v_in_km_per_s = np.sqrt(Ei_in_meV / 5.277)
    average_total_distance = 25 + sum(sample_to_detection_in_m_f)/len(sample_to_detection_in_m_f)
    tof_in_us = ((average_total_distance * 1e-3)) / (v_in_km_per_s*1e-6)
    interval = [tof_in_us*1e-6*0.9, tof_in_us*1e-6*1.1]
    clusters_tof_plot(clusters_bus_no_extra_filter, run, interval=interval, label='Without extra filter')
    clusters_tof_plot(clusters_bus, run, interval=interval, label='With extra filter')
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.legend()
    plt.title('Time-of-flight around %.2f meV' % Ei_in_meV)

    
    # Time-of-Flight, full
    plt.subplot(4, 4, 3)
    clusters_tof_plot(clusters_bus_no_extra_filter, run, interval=[0, 0.1], label='Without extra filter')
    clusters_tof_plot(clusters_bus, run, interval=[0, 0.1], label='With extra filter')
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.legend()
    plt.title('Time-of-flight')
    
    
    # Time-of-flight vs distance
    plt.subplot(4, 4, 8)
    plt.xlabel('ToF ($\mu$s)')
    plt.ylabel('Distance (m)')
    plt.title('ToF vs Distance (using extra filter)')
    bins = [200, 200]
    #interval = [[0, 100000], [0, ]]
    plt.hist2d(tof_in_us_f, sample_to_detection_in_m_f,
               bins=bins,
               norm=LogNorm(),
               #vmin=vmin, vmax=vmax,
               #range=ADC_range,
               cmap='jet',
               weights=(1/duration)*np.ones(len(tof_in_us_f)))
    cbar = plt.colorbar()
    cbar.set_label('Counts/s')
    
    # Energy, zoom
    plt.subplot(4, 4, 11)
    plt.xlabel('Energy (meV)')
    plt.ylabel('Counts')
    plt.title('Energy distribution (when assuming elastic scattering)\nzoom')
    bins_E = 100
    energies_f = e_calc.get_energy(tof_in_us_f, sample_to_detection_in_m_f, moderator_to_sample_in_m)
    energies_uf = e_calc.get_energy(tof_in_us_uf, sample_to_detection_in_m_uf, moderator_to_sample_in_m)
    plt.hist(energies_uf, bins=bins_E, histtype='step', label='Without extra filter', range=[Ei_in_meV*0.9, Ei_in_meV*1.1])
    plt.hist(energies_f, bins=bins_E, histtype='step', label='With extra filter', range=[Ei_in_meV*0.9, Ei_in_meV*1.1])
    plt.yscale('log')
    plt.legend()
    
    # Energy, full
    plt.subplot(4, 4, 12)
    plt.xlabel('Energy (meV)')
    plt.ylabel('Counts')
    plt.title('Energy distribution (when assuming elastic scattering)\nfull')
    bins_E = 100
    energies_f = e_calc.get_energy(tof_in_us_f, sample_to_detection_in_m_f, moderator_to_sample_in_m)
    energies_uf = e_calc.get_energy(tof_in_us_uf, sample_to_detection_in_m_uf, moderator_to_sample_in_m)
    plt.hist(energies_uf, bins=bins_E, histtype='step', label='Without extra filter', range=[0, 50])
    plt.hist(energies_f, bins=bins_E, histtype='step', label='With extra filter', range=[0, 50])
    plt.yscale('log')
    plt.legend()

    # Save data
    fig.set_figwidth(19)
    fig.set_figheight(16)
    plt.tight_layout()

# ==============================================================================
#                               HELPER FUNCTIONS
# ==============================================================================

def mkdir_p(my_path):
    """
    Creates a directory, equivalent to using mkdir -p on the command line.

    Args:
        my_path (str): Path to where the new folder should be created.

    Yields:
        A new folder at the requested path.
    """
    from errno import EEXIST
    from os import makedirs,path
    try:
        makedirs(my_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(my_path):
            pass
        else: raise