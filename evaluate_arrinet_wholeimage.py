# -*- coding: utf-8 -*-
"""
evaluate_arrinet_wholeimage.py

Evaluate ARRInet classification performance on whole tissue image

Created on Thu Apr  2 11:28:38 2020

@author: George Liu

Dependencies: mat, tile_data_ariraw_GSL, arrinet_classify
"""

import numpy as np
import mat
import os
import re
import pandas as pd
import matplotlib.image as mpimg
import tile_data_ariraw_GSL
import arrinet_classify

NAME_PERICHONDRIUM = 'PerichondriumWCartilage'
NAME_CARTILAGE = 'Cartilage' 
PATH_DATASPLIT_KEY = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/arritissue_sessions_key.csv'
PATH_DATA = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/'
PATH_MASK = os.path.join(PATH_DATA, 'masks/')


def main():
    #User select a whole raw image (.mat)
    print('Select an unprocessed, multispectral whole image (.mat):')
    path_image = mat.uigetfile(initialdir='C:/Users/CTLab/Documents/George/Python_data/arritissue_data/', title='Select file', filetypes=(("MAT-file","*.mat"),("all files","*.*")))
    print(path_image)
    
    # Read in image to python workspace
    im = tile_data_ariraw_GSL.import_ariraw_matlab(path_image) # 21 channel multispectral image, float64 ndarray shape (H, W, C) 
    
    # Read in acquisition date
    filename = os.path.basename(path_image)
    date = filename[0:8]
    
    # Read in true tissue class
    pattern = "-(.*?)_"
    this_tissue = re.search(pattern, filename).group(1)
    # relabel true class as cartilage if tissue type is perichondrium
    if this_tissue.lower() == NAME_PERICHONDRIUM.lower():
        true_class = NAME_CARTILAGE
    else:
        true_class = this_tissue
    
    # Read in which dataset the image belongs to (train, validation, or test)
    datasplit_key = pd.read_csv(PATH_DATASPLIT_KEY)
    boolcol_session = datasplit_key["session"]==int(date)
    myrow = [i for i, x in enumerate(boolcol_session) if x]
    dataset = datasplit_key["Split"][myrow[0]]
    
    # Tile image, including foreground and background
    # load mask
    segmentation_folder = os.path.join(PATH_MASK, str(date))
    segmentation_filename = [f for f in os.listdir(segmentation_folder) if f.lower().endswith((this_tissue + 'Mask.png').lower())]
    assert len(segmentation_filename)>0, "Number of segmentation files is: " + str(len(segmentation_filename)) + " in folder " + str(segmentation_folder) + " for tissue " + str(this_tissue)
    segmentation = mpimg.imread(os.path.join(segmentation_folder, str(segmentation_filename[0])))
    # Tile image into 32x32 pixel patches
    list_tiles, list_fracmask_intile, list_loc = tile_data_ariraw_GSL.gettiles3d(im, segmentation, tile_size=(32,32), fracinmask=0)
    stack_tiles = np.stack(list_tiles, axis=0) # 4-D ndarray of shape (N=1980, H=32, W=32, C=21)
    stack_tiles = np.transpose(stack_tiles, axes=(0, 3, 1, 2)) # permute dimensions from ndarray (N, H, W, C) to (N, C, H, W) 

    # Obtain multispectral classification scores and predictions for all tiles in whole image
    class_scores, class_prob, pred_class_int, pred_class_name = arrinet_classify.classify(stack_tiles, isprocessed=False, ismultispectral=True)
    
    # Obtain non-multispectral classification scores and predictions
    class_scores_RGB, class_prob_RGB, pred_class_int_RGB, pred_class_name_RGB = arrinet_classify.classify(stack_tiles[:,:3,:,:], isprocessed=False, ismultispectral=False)

    print('Done.')
    
if __name__=='__main__':
    main()
