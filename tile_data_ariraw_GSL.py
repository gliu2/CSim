# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:27:56 2020

Manually tile ARIRAW image into 32x32 square patches. Tile only portion of image contained within segmentation mask.
Alternative to dataloading_arriGSL.py

Before running:
    -Update CSV file with latest tissue dataset: arritissue_sessions.csv (C:/Users/CTLab/Documents/George/Python_data/arritissue_data)

@author: CTLab
Last edit: 1/28/2020 - adapted from tile_data_GSL.py to import .mat data of ARIRAW image RGB demosaiced unprocessed images.
George Liu

Dependencies: dataloading_arriGSL.py, mat.py
"""

from __future__ import print_function, division
import os
import numpy as np
import mat
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import imageio
import pickle
import hdf5storage # hdf5storage.loadmat(filepath)  function reads in .mat file into python variable space

PATH_DATA = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/'
PATH_MASK = os.path.join(PATH_DATA, 'masks/')
PATH_IM = os.path.join(PATH_DATA, 'arriraw_data/')
#PATH_CSV = os.path.join(PATH_DATA, 'arritissue_sessions.csv') # tile training dataset
#PATH_CSV = os.path.join(PATH_DATA, 'arritissue_sessions_test.csv') # tile test dataset
PATH_CSV = os.path.join(PATH_DATA, 'arritissue_sessions_val.csv') # tile validation dataset
PATH_PATCHES2D = os.path.join(PATH_DATA, 'arriraw_data/patches2d/')
PATH_PATCHES3D = os.path.join(PATH_DATA, 'arriraw_data/patches3d/')

# Fraction of tissue in mask 
MY_FRACINMASK = 1


#%% 8-14-19: Define function to tile image into patches

def gettiles3d(im, mask, tile_size=(32,32), fracinmask=0.5):
    """Composes several transforms together.

    Args:
        im (3D numpy array with dims): image of shape (H, W, Channels)
        mask (3D numpy array): logical array of where tissue is, of shape (H, W, Channels)
        tile_size (2D tuple): size of tile
        fracinmask (double): minimum fraction of mask that must be included in tile
    
    Output:
        cache_tiles (list): tiles of image in mask
    """ 
        
    # Initialize sizes of image and tile    
    (height, width, channels) = np.shape(im)
    (tile_height, tile_width) = tile_size
    
    # Tile indices
    (n_y, n_x) = np.floor(np.divide(np.shape(im)[:2], tile_size))
    n_y = int(n_y)
    n_x = int(n_x)
    
    # Iterate through tiles
    cache_tiles = []
    for i in range(n_y):
        yloc = int(i*tile_size[0]) # top-left corner
        for j in range(n_x):
            xloc = int(j*tile_size[1])
            tile = im[yloc:yloc+tile_height, xloc:xloc+tile_width, :]
            
            # Keep tile if in mask (enough)
            if np.average(mask[yloc:yloc+tile_height, xloc:xloc+tile_width, 0]) >= fracinmask:
                cache_tiles.append(tile)
            
    return cache_tiles

#%% Utilities
def import_ariraw_matlab(filename):
    """
    Import ariraw images, from .mat files by import_ari2mat_GSL.m (Matlab script). Multispectral image is a 21 channel image. 
    
    Output: ndarray
    """ 
    ariraw_im = hdf5storage.loadmat(filename)
    ariraw_RGBim = ariraw_im['arriRGB_21channels']
    return ariraw_RGBim

def save_tilesRGB(list_tiles, target_folder, session_date, this_tissue):
    """
    Save 3-channel, non-multispectral image
    
    Output: pickle file
    """ 
    for i, tile in enumerate(list_tiles):
        filename = str(session_date) + '_' + str(this_tissue) + '_' + str(i) + '.pkl'
        pickle.dump(tile[:,:,:3], open(os.path.join(target_folder, this_tissue, filename), "wb" ))
        
def save_tilesMS(list_tiles, target_folder, session_date, this_tissue):
    """
    Save 21-channel, multispectral image 
    
    Output: pickle file
    """ 
    for i, tile in enumerate(list_tiles):
        filename = str(session_date) + '_' + str(this_tissue) + '_' + str(i) + '.pkl'
        pickle.dump(tile, open(os.path.join(target_folder, this_tissue, filename), "wb" ))


#%% Main function
def main():
   
    # PANDAS indexing of tissues by acqusition session, skipping missing tissue samples not acquired in some sessions
    sessions_frame = pd.read_csv(PATH_CSV)
    root_dir = PATH_IM
    mask_dir = PATH_MASK
    target_folder2d = PATH_PATCHES2D
    target_folder3d = PATH_PATCHES3D

    num_images = np.count_nonzero(sessions_frame.iloc[:,1:].values)
    
    istissue = sessions_frame.iloc[:,1:].values
    sub_row, sub_col = np.nonzero(istissue)
    
    for idx in range(num_images):
        this_row = sub_row[idx]
        this_col = sub_col[idx]
        this_session = sessions_frame.iloc[this_row, 0]
        this_tissue = sessions_frame.columns[this_col + 1]
        
        print('Working on image ' + str(idx) + ' out of ' + str(num_images) + ': ' + str(this_session) + ' ' + this_tissue)
        
        # Load image
        this_folder = os.path.join(root_dir, str(this_session))
        this_mats = [f for f in os.listdir(this_folder) if f.endswith('.mat')]
        image = np.nan
        for k in range(len(this_mats)):
            mat_filename = this_mats[k]
            if this_tissue in mat_filename:
                image = import_ariraw_matlab(os.path.join(this_folder, mat_filename))
                break
                
        # load mask
        segmentation_folder = os.path.join(mask_dir, str(this_session))
        segmentation_filename = [f for f in os.listdir(segmentation_folder) if f.lower().endswith((this_tissue + 'Mask.png').lower())]
        assert len(segmentation_filename)>0, "Number of segmentation files is: " + str(len(segmentation_filename)) + " in folder " + str(segmentation_folder) + " for tissue " + str(this_tissue)
        segmentation = mpimg.imread(os.path.join(segmentation_folder, str(segmentation_filename[0])))
#        segmentation = segmentation[:,:,0] # 2-dim mask of 0 or 1 

        # Tile image in mask
        tiles = gettiles3d(image, segmentation, fracinmask=MY_FRACINMASK)
        # Save RGB 3-channel image (TIFF)
#        print('Saving to folder: ' + target_folder2d)
        save_tilesRGB(tiles, target_folder2d, this_session, this_tissue)
        # Save multispectral 21-channel image (pickle)
#        print('Saving to folder: ' + target_folder3d)
        save_tilesMS(tiles, target_folder3d, this_session, this_tissue)
        
    print('Done')


if __name__=='__main__':
    main()
#    filepath = mat.uigetfile()
#    ariraw_im = import_ariraw_matlab(filepath) 
