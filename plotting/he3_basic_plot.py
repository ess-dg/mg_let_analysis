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
import plotly.io as pio

import plotting.common_plot as cmplt

import calculations.normalization as norm

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
    
    # Plot pixel delimiters
    b_traces = []
    pixel_radius = 0.0254/2 # meters
    tube_length = 4 # meters
    pixels_per_tube = 256
    pixel_height = tube_length / pixels_per_tube
    #for x_c, y_c, z_c in zip(pixels['x'][5:10], pixels['y'][5:10], pixels['z'][5:10]):
    #    # Get limits in cartesian coordinates
    #    zx_angle = norm.get_zx_angle(x_c, z_c)
    #    y_0, y_1 = y_c - pixel_height/2, y_c + pixel_height/2
    #    z_0, z_1 = z_c + pixel_radius*np.cos(zx_angle - np.pi/2), z_c + pixel_radius*np.cos(zx_angle + np.pi/2)
    #    x_0, x_1 = x_c + pixel_radius*np.sin(zx_angle - np.pi/2), x_c + pixel_radius*np.sin(zx_angle + np.pi/2)
    #    # Plot pixel cells
    #    xs = np.array([x_0, x_0, x_1, x_1, x_0])
    #    ys = np.array([y_0, y_1, y_1, y_0, y_0])
    #    zs = np.array([z_0, z_0, z_1, z_1, z_0])
    #    b_trace = go.Scatter3d(x=xs,
    #                           y=ys,
    #                           z=zs,
    #                   mode='line',
    #                   line = dict(
    #                                color='rgba(0, 0, 0, 0.5)',
    #                                width=5)
    #                    )
    #    b_traces.append(b_trace)
    #    # Plot corners of pixel cells based on converted to polar
    #    phi_0, phi_1 = norm.get_phi(x_0, y_0), get_phi(x_1, y_1)
    #    theta_0, theta_1 = norm.get_theta(x_0, y_0, z_0), norm.get_theta(x_1, y_1, z_1)
    #    for phi in [phi_0, phi_1]:
    #        for theta in [theta_0, theta_1]:
    #            b_trace = go.Scatter3d(x=xs,
    ##                                   y=ys,
     #                                  z=zs,
     #                                  mode='markers',
     #                                  line = dict(
     #                                           color='rgba(0, 0, 0, 0.5)',
     #                                           width=5)
     #                                   )
                
    
    #py.offline.init_notebook_mode()
    if region is None:
        data = [trace_1, trace_2]
    else:
        data = [trace_1, trace_2, trace_3]
    for b_trace in b_traces:
        data.append(b_trace)
    fig = go.Figure(data=data, layout=layout)
    py.offline.plot(fig,
                    filename='../output/he3_pixels_location.html',
                    auto_open=True)


# ==============================================================================
#                              PLOT 3D HISTOGRAM
# ==============================================================================

def he3_plot_3D_histogram(df, mapping, region_edges=None, label='', file_name=''):
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
            cmin=0,
            cmax=4,
            colorscale='Jet',
            opacity=1,
            colorbar=dict(thickness=20,
                          title='counts',
                          tickvals=[0, 1, 2, 3, 4],
                          ticktext=['1', '1e1', '1e2', '1e3', '1e4']
                          ),
        ),
        name='Helium-3 tubes',
        text=labels[hist_non_zero_indices],
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
                        title='z (m)'))
    )
    py.offline.init_notebook_mode()
    trace_1.marker.colorbar.x = 0.8
    #trace_1.marker.colorbar.y = 0.65
    #trace_1.marker.colorbar.len = 1#0.3
    if region_edges is not None:
        data = [trace_1, trace_2, trace_3]
    else:
        data = [trace_1]#, trace_3]
    fig = go.Figure(data=data, layout=layout)
    
    annots =  [dict(x=0.5, y=0.95, text=label, showarrow=False, font=dict(size=20))]

    # plot figure
    fig['layout']['annotations'] = annots
    #fig.update_layout(legend=dict(
    #    yanchor="top",
    #    y=0.8,
    #    xanchor="left",
    #    x=0.3,
    #    bgcolor="White",
    #    bordercolor="Black",
    #    borderwidth=2
    #))
    fig.layout.showlegend = False
    #py.offline.iplot(fig)
    py.offline.plot(fig,
                    filename='../output/he3_hist_%s.html' % file_name,
                    auto_open=False)
    pio.write_image(fig, '../output/he3_hist_%s.png' % file_name, scale=2)
    

def he3_plot_3D_histogram_v2(df, mapping, region_edges=None):
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
    weights = df.iloc[:, :].sum(axis=1).to_numpy()
    indices = df.index.values
    number_bins = len(df.index.values)
    hist, bins = np.histogram(indices, bins=number_bins, weights=weights)
    hist_non_zero_indices = np.where(hist != 0)[0]
    labels = []
    for weight, index in zip(weights, indices):
        labels.append('Counts: %d<br>ID %d' % (weight, index))
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
        data = [trace_1, trace_3]
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))
    #py.offline.iplot(fig)
    py.offline.plot(fig,
                    filename='../output/he3_hist.html',
                    auto_open=True)


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
