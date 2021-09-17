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

# ==============================================================================
#                     HELIUM-3: PLOT FIGURE-OF-MERITS
# ==============================================================================

def plot_figure_of_merits(df, number_bins, mapping, region_of_interest,
                          moderator_to_sample_in_m, run, prominence=1e4):
    # Filter data
    df_filtered = df[df['pixel_id'].isin(region_of_interest)]
    # Get figure-of-merits
    values = cmplt.get_all_foms(df_filtered['tof'],
                                mapping['r'][df_filtered['pixel_id']],
                                number_bins,
                                moderator_to_sample_in_m,
                                prominence=prominence,
                                run=run)
    Eis, foms, peak_areas, shoulder_areas = values
    # Plot figure-of-merits
    fig = plt.figure()
    plt.plot(Eis, foms, 'o', color='black')
    plt.xlabel('Energy (meV)')
    plt.ylabel('fom')
    plt.grid(True, which='major', linestyle='--', zorder=0)
    plt.grid(True, which='minor', linestyle='--', zorder=0)
    plt.title('Figure-of-merit vs incident neutron energy')
    plt.tight_layout()
