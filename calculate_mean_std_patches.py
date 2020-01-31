# -*- coding: utf-8 -*-
"""
calculate_mean_std_patches.py
Created on Wed Jan 29 16:28:45 2020

Calculate per-channel mean and standard deviation of training dataset of unprocessed ariraw data.
Before running: import data from .mat -> pickle files using tile_data_ariraw_GSL.py

See for concept:
    http://cs231n.github.io/neural-networks-2/#datapre

Output:
    mean_channel
    std_channel 

@author: CTLab
1-29-2020
"""

import numpy as np
import mat
import os
import pickle

# Hard-coded values
N_CHANNELS = 21

# user select path
#traindata_dir = mat.uigetdir() # choose training folder of path images, with subdirs by tissue type containing patches
traindata_dir = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/arriraw_data/patches3d/train'

# Find all subdirectories
tissue_subdirs = [x[0] for x in os.walk(traindata_dir)]
tissue_subdirs = tissue_subdirs[1:] # remove parent directory, to include only subdirectories

# initialize mean and std channel values
mean_channel = np.zeros(N_CHANNELS)
std_channel = np.zeros(N_CHANNELS)
total_patches = 0

# Calculate mean and std per channel, across all tissue types, in training tiles
n_tissues = len(tissue_subdirs)
for i in range(n_tissues):
    print('Working on ' + str(i) + ' out of ' + str(n_tissues) + ' tissues....')
    this_folder = tissue_subdirs[i]
    patches = os.listdir(this_folder)
    num_patches = len(patches)
    
    for j in range(num_patches):
        patch_pickled = patches[j]
        patch = pickle.load( open( os.path.join(this_folder, patch_pickled), "rb" ) )
        mean_channel += np.mean(patch, axis=(0,1)) 
        std_channel += np.std(patch, axis=(0,1))
    
    total_patches += num_patches
    
mean_channel /= total_patches
std_channel /= total_patches

print('Done.')