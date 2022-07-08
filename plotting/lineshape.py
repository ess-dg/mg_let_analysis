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

import calculations.energy as e_calc
import calculations.fitting as fit

import plotting.common_plot as cmplt


# ==============================================================================
#                              GET FIGURE-OF-MERIT
# ==============================================================================

def get_figure_of_merit(Ei_in_meV, tof_in_us, sample_to_detection_in_m,
                        moderator_to_sample_in_m):
    
    # Get event-by-event information on energy transfer
    delta_E = e_calc.get_energy_transfer(Ei_in_meV, tof_in_us,
                                         sample_to_detection_in_m,
                                         moderator_to_sample_in_m)
    
    
    