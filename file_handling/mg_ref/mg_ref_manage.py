#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
manage.py: Contains functions to manage Multi-Grid data
"""

import numpy as np
import pandas as pd

import file_handling.mg_ref.mg_ref_read as mg_read

# ==============================================================================
#                                  FILTER DATA
# ==============================================================================

def filter_data(df, parameters):
    """
    Filters clusters based on preferences set on GUI.

    Args:
        df (DataFrame): Clustered events
        parameters (dict): Dictionary containing information on which
                           parameters to filter on, and within which range.

    Returns:
        df_red (DataFrame): DataFrame containing the reduced data according to
                            the specifications in "parameters".
    """

    df_red = df
    for parameter, (min_val, max_val, filter_on) in parameters.items():
        if filter_on:
            if parameter == 'layer':
                df_red = df_red[((df_red.wch % 20) >= min_val) &
                                ((df_red.wch % 20) <= max_val)]
            elif parameter == 'row':
                df_red = df_red[(((df_red.bus * 4) + df_red.wch//20) >= min_val) &
                                (((df_red.bus * 4) + df_red.wch//20) <= max_val)]
            else:
                df_red = df_red[(df_red[parameter] >= min_val) &
                                (df_red[parameter] <= max_val)]
    return df_red

# ==============================================================================
#                                MERGE FILES
# ==============================================================================

def merge_files(dfs):
    """
    Function to merge DataFrames containing Multi-Grid data. When subsequent
    files are appended, the timestamps are updated on each file such that the
    first file has the time=0 reference.

    Args:
        dfs (np.array): Array of DataFrames containing the Multi-Grid data

    Returns:
        df_full (pd.DataFrame): DataFrame containing appended Multi-Grid data

    """

    # Declare first element and get max time
    df_full = dfs[0]
    start = df_full.shape[0]
    max_time = df_full.tail(1)['time'].array[0]
    # Append all files so that subsequent files get an increased timestamp
    for df in dfs[1:]:
        df_full = df_full.append(df, ignore_index=True)
        df_full['time'].loc[start:] = df_full['time'].loc[start:] + max_time
        max_time = df_full.tail(1)['time'].array[0]
        start = df_full.shape[0]
    return df_full

# ==============================================================================
#                             EXTRACT AND SAVE DATA
# ==============================================================================

def extract_and_save(file_name, raw_folder, processed_folder):
    """
    Function to extract, cluster and save data.

    Args:
        file_name (str): Name of file
        raw_folder (str): Path to the folder with the raw data
        processed_folder (str): Location of clusters and events

    Yields:
        Clusters and events are extracted from the raw data and saved in the
        'processed'-folder.

    """
    raw_path = raw_folder + file_name + '.zip'
    clusters_path = processed_folder + file_name + '_clu.h5'
    events_path = processed_folder + file_name + '_ev.h5'
    extract_data(raw_path, clusters_path, events_path)


# ==============================================================================
#                            LOAD CLUSTERS AND EVENTS
# ==============================================================================

def load_clusters_and_events(file_name, processed_folder):
    """
    Function to load data from a specific run.

    Args:
        file_name (str): File name
        processed_folder (str): Location of clusters and events

    Returns:
        Clusters (DataFrame)
        Events (DataFrame)

    """
    clusters_path = processed_folder + file_name + '_clu.h5'
    events_path = processed_folder + file_name + '_ev.h5'
    return load_data(clusters_path), load_data(events_path)


# ==============================================================================
#                             HELPER FUNCTIONS
# ==============================================================================

def extract_data(zipped_path, clusters_save_path, events_save_path):
    """
    Function to extract, cluster and save data.

    Args:
        zipped_path (str): Location of raw data
        clusters_save_path (str): Destination for clusters
        events_save_path (str): Destination for events

    Yields:
        Clusters and events are extracted from the raw data and saved at the
        specified locations

    """
    unzipped_path = mg_read.unzip_data(zipped_path)
    data = mg_read.import_data(unzipped_path)
    # Extract clusters and save to disc
    clusters = mg_read.extract_clusters(data)
    save_data(clusters, clusters_save_path)
    clusters = None
    # Extract events and save to disc
    events = mg_read.extract_events(data)
    save_data(events, events_save_path)
    events = None
    # Clear data
    data = None

def save_data(df, path):
    """
    Saves clusters or events to specified HDF5-path.

    Args:
        path (str): Path to HDF5 containing the saved DataFrame
        df (DataFrame): Data

    Yields:
        Data is saved to path
    """
    # Export to HDF5
    df.to_hdf(path, 'df', complevel=9)


def load_data(path):
    """
    Loads clustered data from specified HDF5-path.

    Args:
        path (str): Path to HDF5 containing the saved data

    Returns:
        df (DataFrame): Data
    """
    df = pd.read_hdf(path, 'df')
    return df
