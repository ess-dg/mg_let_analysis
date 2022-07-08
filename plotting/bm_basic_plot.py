#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bm_basic_plot.py: Contains the basic functions to plot beam monitor data
"""
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd



# ==============================================================================
#                               TOF HISTOGRAM
# ==============================================================================

def bm_plot_basic(df, number_bins, run):
    """
    Function to plot the tof histograms from the different beam monitors

    Args:
        df (pd.DataFrame): Beam monitor data
        number_bins (int): The number of bins to split the histogram into
        run (str): Run title of the measurement

    Yields:
        Time-of-flight histograms from the different beam monitors
    """
    fig = plt.figure()
    dirname = os.path.dirname(__file__)
    output_path = os.path.join(dirname, '../output/%s_energy_histogram.png' % run)
    fig.suptitle('LET beam monitor time-of-flight histogram (%s)' % run)
    for i in np.arange(0, 8, 1):
        plt.subplot(2, 4, i+1)
        df_bm = df.loc[i]
        weights = df_bm.values
        plt.xlabel('tof (µs)')
        plt.xlim(0, 100000)
        plt.ylabel('counts')
        plt.title('Monitor %d' % (i))
        if sum(weights) > 0:
            hist, bins, __ = plt.hist(df_bm.index, weights=weights,
                                      label='Monitor %d' % (i), zorder=5,
                                      histtype='step', range=[0, 100000],
                                      bins=number_bins, color='black')
            bin_centers = (bins[:-1] + bins[1:]) / 2
            plt.grid(True, which='major', linestyle='--', zorder=0)
            plt.grid(True, which='minor', linestyle='--', zorder=0)
            plt.yscale('log')
            # Save as text_file
            txt_path = os.path.join(dirname,
                                    '../output/bm_%d_run_%s.txt' % (i+1, run))
            np.savetxt(txt_path,
                       np.transpose(np.array([bin_centers, hist, np.sqrt(hist)])),
                       delimiter=",",header='tof (us), counts, error')
    # Save data
    fig.set_figwidth(14)
    fig.set_figheight(7)
    plt.tight_layout()
    plt.show()
    plot_path = os.path.join(dirname,
                             '../output/bm_run_%s_tof_histogram.png' % run)
    fig.savefig(output_path, bbox_inches='tight')
