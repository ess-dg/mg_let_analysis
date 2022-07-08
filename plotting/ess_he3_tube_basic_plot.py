#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
he3_basic_plot.py: Contains the basic functions to plot data taken with the ESS helium-3 tube.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import file_handling.ess_he3_tube.read as ess_he3_tube_read

def plot_he3(df, sup_title, he3_filter, he3_area, adc_bins=100, time_bins=50):
    # Filter data
    df_f = ess_he3_tube_read.filter_data(df, he3_filter)
    # Plot data
    fig = plt.figure()
    fig.set_figwidth(12)
    fig.set_figheight(8)
    plt.suptitle(sup_title, fontsize=15, fontweight='bold', y=0.97)
        
    plt.subplot(2, 2, 2)
    plt.title('Rate vs Time')
    # Prepare histogram
    time = (df.tof)/(60 ** 2)
    hist, bin_edges = np.histogram(time, bins=time_bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    delta_t = (60 ** 2) * (bin_centers[1] - bin_centers[0])
    # Get rate
    number_events = len(df.tof)
    duration = df.tof.values[-1] - df.tof.values[0]
    average_rate_per_m2 = number_events/(duration*he3_area)
    average_rate_per_m2_error = np.sqrt(number_events)/(duration*he3_area)
    # Plot
    plt.errorbar(bin_centers, (hist/delta_t), np.sqrt(hist)/delta_t, marker='.', linestyle='',
                 zorder=5, color='blue', label='OFF')
    # Prepare histogram
    time_f = (df_f.tof)/(60 ** 2)
    hist_f, bin_edges_f = np.histogram(time_f, bins=time_bins)
    bin_centers_f = 0.5 * (bin_edges_f[1:] + bin_edges_f[:-1])
    delta_t_f = (60 ** 2) * (bin_centers_f[1] - bin_centers_f[0])
    # Get rate
    number_events_f = len(df_f.tof)
    duration_f = df_f.tof.values[-1] - df_f.tof.values[0]
    average_rate_per_m2_f = number_events_f/(duration_f*he3_area)
    average_rate_per_m2_error_f = np.sqrt(number_events_f)/(duration_f*he3_area)
    #Plot
    plt.errorbar(bin_centers_f, (hist_f/delta_t_f), np.sqrt(hist_f)/delta_t_f, marker='.', linestyle='',
                 zorder=5, color='red', label='ON')
    plt.legend(title='Filter')
    plt.xlabel('Time (hours)')
    plt.ylabel('Rate (Hz)')
    plt.yscale('log')
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    
    plt.subplot(2, 2, 1)
    plt.title('Channel histogram')
    a_rate = average_rate_per_m2
    a_error = average_rate_per_m2_error
    a_rate_f = average_rate_per_m2_f
    a_error_f = average_rate_per_m2_error_f
    plt.hist(df.ch, histtype='step', color='blue', label='OFF (Rate: %.2f ± %.2f Hz/m$^2$)' % (a_rate, a_error),
             zorder=5, bins=4)
    plt.hist(df_f.ch, histtype='step', color='red', label='ON (Rate: %.2f ± %.2f Hz/m$^2$)' % (a_rate_f, a_error_f),
             zorder=5, bins=4)
    plt.legend(title='Filter')
    plt.xlabel('Channel')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.xlim(0, 4)
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    
    plt.subplot(2, 2, 3)
    plt.title('ADC histogram')
    plt.hist(df.adc, histtype='step', color='blue', label='OFF', zorder=5, range=[0, 66000], bins=adc_bins)
    plt.hist(df_f.adc, histtype='step', color='red', label='ON', zorder=5, range=[0, 66000], bins=adc_bins)
    plt.legend(title='Filter')
    plt.xlabel('Charge (ADC channels)')
    plt.ylabel('Counts')
    plt.xlim(0, 80000)
    plt.yscale('log')
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    
    plt.subplot(2, 2, 4)
    plt.title('Pile-up histogram')
    plt.hist(df.pile_up, histtype='step', color='blue', label='OFF', zorder=5)
    plt.hist(df_f.pile_up, histtype='step', color='red', label='ON', zorder=5)
    plt.legend(title='Filter')
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.xlabel('Pile up (0=no, 1=yes)')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.xlim(0, 1)

    plt.tight_layout()
    output_path = '../output/%s.png' % sup_title
    fig.savefig(output_path, bbox_inches='tight')