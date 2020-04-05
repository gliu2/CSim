# -*- coding: utf-8 -*-
"""
evaluate_arrinet_wholeimage.py

Evaluate ARRInet classification performance on whole tissue image

Created on Thu Apr  2 11:28:38 2020

@author: George Liu
Last edit: 4-4-2020

Dependencies: mat, tile_data_ariraw_GSL, arrinet_classify
"""

import numpy as np
import mat
import os
import re
import pandas as pd
import matplotlib.image as mpimg
from collections import Counter
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

    # Print results
    print('')
    print('Acquisition date:', date)
    print('Dataset:', dataset)
    trueclass_ind = arrinet_classify.class_str2int(true_class)
    print('True class:', true_class, '(', np.average(class_prob, axis=0)[trueclass_ind], 'MS prob,', np.average(class_prob_RGB, axis=0)[trueclass_ind], 'RGB prob)')
    print('--- Using all tiles ---')
    c = Counter(pred_class_name) # most common predicted class in all tiles - multispectral
    pred_class_int_alltiles = np.argmax(np.bincount(pred_class_int.flatten())) # index of most common predicted class in all tiles - multispectral
    c_RGB = Counter(pred_class_name_RGB)
    pred_class_int_alltiles_RGB = np.argmax(np.bincount(pred_class_int_RGB.flatten()))
    print('Predicted multispectral class (average prob):', c.most_common(1), '(', np.average(class_prob, axis=0)[pred_class_int_alltiles], ')')
    print('Predicted RGB class (average prob):', c_RGB.most_common(1), '(', np.average(class_prob_RGB, axis=0)[pred_class_int_alltiles_RGB], ')')
    num_tiles = len(list_tiles)
    num_tiles_allmask = sum(i == 1 for i in list_fracmask_intile)
    print('Number of tiles:', num_tiles)
    
    print('--- Using only 100% mask tiles ---')
    print('Number of tiles filled:', num_tiles_allmask)
    

    print('Done.')
    
if __name__=='__main__':
    main()
