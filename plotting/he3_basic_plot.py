#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
he3_basic_plot.py: Contains the basic functions to plot LET helium-3 data.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from plotly.offline import iplot, plot
import plotly as py
import plotly.graph_objects as go

import plotting.common_plot as cmplt

# ==============================================================================
#                               PLOT COORDINATES
# ==============================================================================

def he3_plot_coordinates(pixels, region=None):
    """
    Function to plot the positions of all the pixels in the LET helium-3 array,
    as well as highlighting a region of interest.

    Args:
        pixels (np.array): Pixel coordinates in Helium-3 array
        region (np.array): Pixel ID's corresponding to a small region in the
                           helium-3 array which is to be highlighted.

    Yields:
        A 3D plot showing all the pixels in the helium-3 array (blue), as well
        highlightning the position of the pixel id's provided in 'region'
        (green), and showing the sample position (red).

    """

    # Define labels
    labels = []
    for i in np.arange(0, len(pixels['x']), 1):
        labels.append('ID: %d<br>θ: %.2f°<br>φ: %.2f°' % (i, pixels['theta'][i],
                                                          pixels['phi'][i]))
    trace_1 = go.Scatter3d(
        x=pixels['x'],
        y=pixels['y'],
        z=pixels['z'],
        mode='markers',
        marker=dict(
            size=4,
            line=dict(
                color='rgba(0, 0, 255, 1)',
                width=1.0
            ),
            opacity=1.0
        ),
        name='Helium-3 tubes',
        text=labels
    )

    trace_2 = go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        marker=dict(
            size=4,
            line=dict(
                color='rgba(255, 0, 0, 1)',
                width=5.0
            ),
            opacity=1.0
        ),
        name='Sample'
    )

    if region is not None:
        # Define labels
        labels = []
        for i in region:
            labels.append('ID: %d<br>θ: %.2f°<br>φ: %.2f°' % (i, pixels['theta'][i],
                                                              pixels['phi'][i]))
        trace_3 = go.Scatter3d(
                            x=pixels['x'][region],
                            y=pixels['y'][region],
                            z=pixels['z'][region],
                            mode='markers',
                            marker=dict(
                                    size=4,
                                    line=dict(
                                            color='rgba(0, 255, 0, 1)',
                                            width=1.0
                                            ),
                                    opacity=1.0
                            ),
                            name='Multi-Grid sized region',
                            text=labels
                            )



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
                        title='z (m)'),)
    )
    py.offline.init_notebook_mode()
    if region is None:
        data = [trace_1, trace_2]
    else:
        data = [trace_1, trace_2, trace_3]
    fig = go.Figure(data=data, layout=layout)
    py.offline.iplot(fig)


# ==============================================================================
#                              PLOT 3D HISTOGRAM
# ==============================================================================

def he3_plot_3D_histogram(df, mapping, region_edges=None):
    """
    Function to plot a 3D histogram of the hit-positions of neutrons in the
    helium-3 array.

    Args:
        df (pd.DataFrame): Helium-3 data
        mapping (np.array): Helium-3 'pixel_id->(x, y, z)'->mapping
        region (np.array): Pixel ID's corresponding to the edges of a small
                           region in the Helium-3 array which is to be
                           highlighted.

    Yields:
        A 3D histgram showing the distribution of hit-positions on the
        helium-3 array.
    """

    # Histogram data
    hist, __ = np.histogram(df['pixel_id'], bins=len(mapping['x']),
                            range=[0, len(mapping['x'])])
    hist_non_zero_indices = np.where(hist != 0)[0]
    # Define labels
    labels = []
    for i in np.arange(0, len(mapping['x']), 1):
        labels.append('ID: %d<br>Counts: %d<br>θ: %.2f°<br>φ: %.2f°' % (i, hist[i],
                                                                        mapping['theta'][i],
                                                                        mapping['phi'][i]))
    labels = np.array(labels)
    # Plot
    trace_1 = go.Scatter3d(
        x=mapping['x'][hist_non_zero_indices],
        y=mapping['y'][hist_non_zero_indices],
        z=mapping['z'][hist_non_zero_indices],
        mode='markers',
        marker=dict(
            size=4,
            color=np.log10(hist[hist_non_zero_indices]),
            colorscale='Jet',
            opacity=1,
            colorbar=dict(thickness=20,
                          title='log10(counts)'
                          ),
        ),
        name='Helium-3 tubes',
        text=labels[hist_non_zero_indices]
    )

    if region_edges is not None:
        trace_2 = go.Scatter3d(
                    x=mapping['x'][region_edges]-0.05,
                    y=mapping['y'][region_edges],
                    z=mapping['z'][region_edges]-0.05,
                    mode='lines',
                    line=dict(color='rgba(0, 0, 0, 1)',
                              width=5),
                    name='Region of interest',
                    )

    trace_3 = go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode='markers',
        marker=dict(
            size=4,
            line=dict(
                color='rgba(255, 0, 0, 1)',
                width=5.0
            ),
            opacity=1.0
        ),
        name='Sample'
    )

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
                        title='z (m)'),)
    )
    py.offline.init_notebook_mode()
    if region_edges is not None:
        data = [trace_1, trace_2, trace_3]
    else:
        data = [trace_2, trace_3]
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    py.offline.iplot(fig)


# ==============================================================================
#                              TIME-OF-FLIGHT
# ==============================================================================

def plot_tof(df, number_bins, region_of_interest, run):
    """
    Function to plot a histogram of time-of-flight in the helium-3 tubes,
    comparing data from the full array and the data in the region-of-interest.

    Args:
        df (pd.DataFrame): Helium-3 data
        number_bins (int): The number of bins to split the histogram into
        region_of_interest (np.array): Pixel ID's corresponding to the region in
                                       the helium-3 array which is to be
                                       investigated
        run (str): Run title of the measurement

    Yields:
        A histogram of time-of-flight overlaying the results from the full
        array and the region of interest
    """
    # Filter data
    df_filtered = df[df['pixel_id'].isin(region_of_interest)]
    # Plot
    fig = plt.figure()
    cmplt.tof_histogram_plot(df, number_bins, color='red',
                             run=run, label='Full array')
    cmplt.tof_histogram_plot(df_filtered, number_bins, color='blue',
                             run=run, label='Region of interest')
    plt.legend(title='Data')
    # Save plot
    dirname = os.path.dirname(__file__)
    output_path = os.path.join(dirname, '../output/%s_tof_histogram.png' % run)
    plt.title('Time-of-Flight (%s)' % run)
    fig.savefig(output_path, bbox_inches='tight')


# ==============================================================================
#                                  ENERGY
# ==============================================================================

def plot_energy(df, number_bins, mapping, region_of_interest,
                moderator_to_sample_in_m, run):
    """
    Function to plot a histogram of

    Args:
        df (pd.DataFrame): Helium-3 data
        number_bins (int): The number of bins to split the histogram into
        mapping (np.array): Pixel_id->(x, y, z, r, theta, phi) mapping
        region_of_interest (np.array): Pixel ID's corresponding to the region in
                                       the helium-3 array which is to be
                                       investigated
        moderator_to_sample_in_m (float): Distance between moderator and sample
        run (str): Run title of the measurement

    Yields:
        A histogram of energy overlaying the results from the full
        array and the region of interest
    """
    # Filter data
    df_filtered = df[df['pixel_id'].isin(region_of_interest)]
    # Plot
    fig = plt.figure()
    sample_to_detection = mapping['r'][df['pixel_id']]
    sample_to_detection_filtered = mapping['r'][df_filtered['pixel_id']]
    cmplt.energy_histogram_plot(df['tof'], sample_to_detection,
                                number_bins, moderator_to_sample_in_m,
                                color='red', run=run, label='Full array',
                                interval=[0, 350])
    cmplt.energy_histogram_plot(df_filtered['tof'], sample_to_detection_filtered,
                                number_bins, moderator_to_sample_in_m,
                                color='blue', run=run,
                                label='Region of interest',
                                interval=[0, 350])
    plt.legend(title='Data')
    # Save plot
    dirname = os.path.dirname(__file__)
    output_path = os.path.join(dirname, '../output/%s_energy_histogram.png' % run)
    plt.title('Energy (%s)' % run)
    fig.savefig(output_path, bbox_inches='tight')
