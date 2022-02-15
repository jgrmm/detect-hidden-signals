#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extracting coherence matrices for a long period of DAS data.

NOTE: For large datasets (multiple TBs) this might take several hours to days,
depending on available computing power.

Created on Thu Oct 28 12:23:43 2021

@author: Julius Grimm (ISTerre, Université Grenoble Alpes)
"""

print("Start python script")


# Import modules
print("Importing modules...")
import a1das
import os
import numpy as np
from scipy.signal import hann
import glob
import config
from helper_functions import *
print("Done importing!")


# Specify root directory containing DAS files
ROOT_FOLDER = 'PATH_TO_DAS_DATA'
RESULT_FOLDER = "PATH_TO_RESULT_FOLDER"
files_all = sorted(glob.glob(ROOT + '*.h5*'))[:-1]
nr_files = len(files_all)
print(f"There are {nr_files} files to be processed.")

# Specify short and long window lenghts
fs = 250
ave_window_min = 20  # 20 min
nr_win_per_2h = 2 * 60 // ave_window_min
len_ave_window = ave_window_min * 60 * fs  # 20min x 60 sec x 250 Hz
len_short_window = 120*fs  # 120 sec x 250 Hz

# Set frequencies to retrieve
config.weights = None
fb0 = list(np.linspace(1,10,20))  # 20 frequencies between 1 and 10 Hz
fb1 = list(np.linspace(10,20,20))  # 20 frequencies between 1 and 10 Hz
fb2 = list(np.linspace(20,40,20))  # 20 frequencies between 1 and 10 Hz


# Loop over all files, reading in 2 files at the time. This depends on the size
# of individual files and the available RAM size
for fid in range(nr_files//2):
    # Extracting date and time from filename, specific to city of Grenoble
    # dataset
    date_time = files_native[2*fid].split("/")[-1]
    date_only = date_time.split("_")[1]
    time_only = date_time.split("_")[2]

    # Reading in a single DAS data file
    f = a1das.open(files_native[2*fid],format='febus')
    a11=f.read()
    f.close()

    # Reading in a second (consecutive) DAS data file
    f = a1das.open(files_native[2*fid+1],format='febus')
    a12=f.read()
    f.close()

    # Merging strain-rate data of both files and deleting original object to
    # save memory
    a1 = np.concatenate((a11.data, a12.data))
    del a11
    del a12


    print("Starting processing...")
    # Looping over all long time windows
    for k in range(nr_win_per_2h):
        data_ave_window = a1[k*len_ave_window:(k+1)*len_ave_window,1:-3]
        # Pass data of long window to function computing coherence matrices
        # for specified frequency bands
        cm_ave = coherence_matrix_parallel(data_ave_window[:,:], ws=250*120,
                                           freq_band_list=[fb0,fb1,fb2], fs=fs)

        print("Time: " + date_only + "_" + time_only)
        print("Minute:", k*ave_window_min)
        print("CovMat shape: ", cm_ave.shape)

        # Saving all computed coherence matrices for a single long time window
        # to disc
        np.save(RESULT_FOLDER + date_only + "_" + time_only +
                f"_k={k}", cm_ave)
