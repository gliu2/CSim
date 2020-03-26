# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:40:06 2019

Manually tile image into square patches. Tile only portion of image contained within segmentation mask.
Alternative to dataloading_arriGSL.py

Before running:
    -Update CSV file with latest tissue dataset: arritissue_sessions.csv (C:/Users/CTLab/Documents/George/Python_data/arritissue_data)

@author: CTLab
Past edit: 8-25-19 - updated tile shape from 36x36 (error) -> (32x32). All previous models trained on 36x36 tiles! 
Last edit: 3/26/20 - updated tile shape back to 36x36 to maintain consistency with prior results
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

PATH_DATA = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/'
PATH_MASK = os.path.join(PATH_DATA, 'masks/')
# Tile training data
PATH_IM = os.path.join(PATH_DATA, 'train/')
#PATH_CSV = os.path.join(PATH_DATA, 'arritissue_sessions.csv')
PATH_CSV = os.path.join(PATH_DATA, 'arritissue_sessions_test.csv')
#PATH_PATCHES2D = os.path.join(PATH_DATA, 'train_patches2d/')
#PATH_PATCHES3D = os.path.join(PATH_DATA, 'train_patches3d/')
# Tile val data
#PATH_IM = os.path.join(PATH_DATA, 'val/')
#PATH_CSV = os.path.join(PATH_DATA, 'arritissue_sessions_val.csv')
PATH_PATCHES2D = os.path.join(PATH_DATA, 'patches2d/')
PATH_PATCHES3D = os.path.join(PATH_DATA, 'patches3d/')

myclasses = ["Artery",
"Bone",
"Cartilage",
"Dura",
"Fascia",
"Fat",
"Muscle",
"Nerve",
"Parotid",
"PerichondriumWCartilage",
"Skin",
"Vein"]
num_classes = len(myclasses)

# Fraction of tissue in mask 
MY_FRACINMASK = 1

#%% 8-14-19: Define function to tile image into patches

def gettiles2d(im, mask, tile_size=(36,36), fracinmask=0.5):
    """Composes several transforms together.

    Args:
        im (2D numpy array): image
        mask (2D numpy array): logical array of where tissue is.
        tile_size (tuple): size of tile
        fracinmask (double): minimum fraction of mask that must be included in tile
    
    Output:
        cache_tiles (list): tiles of image in mask
    """ 
        
    # Initialize sizes of image and tile    
    (height, width) = np.shape(im)
    (tile_height, tile_width) = tile_size
    
    # Tile indices
    (n_y, n_x) = np.floor(np.divide(np.shape(im), tile_size))
    n_y = int(n_y)
    n_x = int(n_x)
    
    # Iterate through tiles
    cache_tiles = []
    for i in range(n_y):
        yloc = int(i*tile_size[0]) # top-left corner
        for j in range(n_x):
            xloc = int(j*tile_size[1])
            tile = im[yloc:yloc+tile_height, xloc:xloc+tile_width]
            
            # Keep tile if in mask (enough)
            if np.average(mask[yloc:yloc+tile_height, xloc:xloc+tile_width]) >= fracinmask:
                cache_tiles.append(tile)
            
    return cache_tiles

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
def save_tilesRGB(list_tiles, target_folder, session_date, this_tissue):
    """
    Save 3-channel, non-multispectral image
    
    Output: TIFF image
    """ 
    for i, tile in enumerate(list_tiles):
        filename = str(session_date) + '_' + str(this_tissue) + '_' + str(i) + '.tiff'
        imageio.imwrite(os.path.join(target_folder, this_tissue, filename), tile[:,:,:3])
        
def save_tilesMS(list_tiles, target_folder, session_date, this_tissue):
    """
    Save 21-channel, multispectral image 
    
    Output: pickle file
    """ 
    for i, tile in enumerate(list_tiles):
        filename = str(session_date) + '_' + str(this_tissue) + '_' + str(i) + '.pkl'
        pickle.dump(tile, open(os.path.join(target_folder, this_tissue, filename), "wb" ))

#%% Test scripts
def test_gettiles2d():
    """
    Test gettiles2d()
    """ 
    
    im1 = np.reshape(np.arange(36), (6,6))
    mask = np.array([[ 0,  0,  1,  1,  1,  0],
       [ 0,  0,  8,  9, 0, 0],
       [12, 13, 14, 15, 16, 17],
       [18, 19, 20, 21, 22, 23],
       [0, 0, 26, 27, 0, 0],
       [0, 0, 32, 33, 0, 0]])
    mask = mask>0
    
    tile_size = (2,2)
    im1_tiles = gettiles2d(im1, mask, tile_size)
    print('Original image:')
    print(im1)
    print('Tiles of size ' + str(tile_size) +':')
    print('Num tiles: ' + str(len(im1_tiles)))
    for i in range(len(im1_tiles)):
        print(im1_tiles[i])

def test_gettiles3d():
    print('Image:')
    filepath1 = mat.uigetfile()
    this_image = mpimg.imread(filepath1)
    print('Image dims: ' + str(np.ndim(this_image)))
    print('Image shape: ' + str(np.shape(this_image)))
    
    print('Mask:')
    filepath2 = mat.uigetfile()
    this_mask = mpimg.imread(filepath2)
    print('Mask dims: ' + str(np.ndim(this_mask)))
    print('Mask shape: ' + str(np.shape(this_mask)))
    
    im2_tiles = gettiles3d(this_image, this_mask)
#    im2_tiles = gettiles3d(this_image, np.ones(np.shape(this_image))) # no mask
    
    print('Original image:')
    plt.figure()
    plt.imshow(this_image)
    print('Num tiles: ' + str(len(im2_tiles)))
    for i in range(len(im2_tiles)):
        plt.figure()
        plt.imshow(im2_tiles[i])

#%% Main function
def main():
#    test_gettiles2d()
#    test_gettiles3d()
   
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
        # Concatenate 7 narrow-band 3-channel images (RGBx1) into 1 multispectral 21-channel image (RGBx7)
        this_folder = os.path.join(root_dir, str(this_session), this_tissue)
        this_tiffs = [f for f in os.listdir(this_folder) if f.endswith('.tif')]
        image = np.nan
        for img_name in this_tiffs:
            this_image = mpimg.imread(os.path.join(this_folder, img_name)) # read image
            if np.isnan(np.sum(image)): # no concatenation necessary for 1st image
                image = this_image
            else:
                image = np.concatenate((image, this_image), axis = 2)
                
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
