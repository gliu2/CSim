# -*- coding: utf-8 -*-
"""
evaluate_arrinet_wholeimage.py

Evaluate ARRInet classification performance on whole tissue image

Created on Thu Apr  2 11:28:38 2020

@author: George Liu
Last edit: 4-17-2020

Dependencies: mat, tile_data_ariraw_GSL, arrinet_classify
"""

import numpy as np
import mat
import os
import re
import pandas as pd
from collections import Counter
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import cm
import tile_data_ariraw_GSL
import arrinet_classify
import matplotlib as mpl
import seaborn as sns
import pickle
from copy import deepcopy
import sklearn
import sklearn.metrics
import numpy.matlib
from scipy import interp
from itertools import cycle

NAME_PERICHONDRIUM = 'PerichondriumWCartilage'
NAME_CARTILAGE = 'Cartilage' 
PATH_DATASPLIT_KEY = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/arritissue_sessions_key.csv'
PATH_DATA = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/'
PATH_MASK = os.path.join(PATH_DATA, 'masks/')
PATH_OUT = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/arriraw_data/arrinet_eval_output/'
PATH_EVAL_OUTPUT = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/arriraw_data/arrinet_eval_output/'

NUM_CLASSES = 11

TISSUE_CLASSES = ["Background", "Artery",
"Bone",
"Cartilage",
"Dura",
"Fascia",
"Fat",
"Muscle",
"Nerve",
"Parotid",
"Skin",
"Vein"]

TISSUE_CLASSES_FINAL = ["Artery",
"Bone",
"Cartilage",
"Dura",
"Fascia",
"Fat",
"Muscle",
"Nerve",
"Parotid",
"Skin",
"Vein"]

MEAN_CHANNEL_PIXELVALS = np.array([
1125.36,
1583.53,
860.107,
1.47882,
1.99471,
13.4813,
3.31147,
37.9476,
5.82629,
75.2963,
45.4425,
43.1282,
20.8992,
1.39119,
1.01157,
0.813617,
0.872862,
0.913199,
67.9463,
64.1412,
18.6965,
])

def cmap_set0gray(base_cmap):
    """Obtain colormap with zero set to gray
    Source: https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar
    """

    cmap = plt.cm.get_cmap(base_cmap)  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    cmaplist[0] = (.5, .5, .5, 1.0)
    
    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    return cmap

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map
    Source: https://gist.github.com/jakevdp/91077b0cae40f8f8244a
    """

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def tiles2im(list_tile_values, mask, image_size=(1080, 1920), tile_size=(32,32), fracinmask=0.5):
    """Quilt together list of tile values to make big image. 
    Reverse of tile_data_ariraw_GSL.gettiles3d(...) function

    Args:
        list_tile_values (numpy array): array of values of each tile, shape (N, 1)
        mask (3D numpy array): logical array of where tissue is, of shape (H, W, Channels)
        image_size (2D tuple): spatial dimensions of big image
        tile_size (2D tuple): size of tile
        fracinmask (double): minimum fraction of mask that must be in tile to copy it
    
    Output:
        allclass_heatmap (2D numpy array): grayscale big image heatmap of tile values quilted together
    """ 
        
    # Initialize sizes of image and tile    
    (height, width) = image_size[:2]
    (tile_height, tile_width) = tile_size
    
    # Tile indices
    (n_y, n_x) = np.floor(np.divide((height, width), tile_size))
    n_y = int(n_y)
    n_x = int(n_x)
    
    # Iterate through tiles
    count = 0
    allclass_heatmap = np.zeros((height, width))
    for i in range(n_y):
        yloc = int(i*tile_size[0]) # top-left corner
        for j in range(n_x):
            xloc = int(j*tile_size[1])
            allclass_heatmap[yloc:yloc+tile_height, xloc:xloc+tile_width] = list_tile_values[count]
            
            # Keep tile if in mask (enough)
            frac_intile = np.average(mask[yloc:yloc+tile_height, xloc:xloc+tile_width, 0])
            if frac_intile < fracinmask:
                allclass_heatmap[yloc:yloc+tile_height, xloc:xloc+tile_width] = 0
                
            count = count + 1
                
            
    return allclass_heatmap

def evaluate_bigimage(path_image):
    """Evaluate Arrinet on whole raw image, using multispectral and non-multispectral data.
    Saves figures of TIFF image and heatmaps of class predictions and true class probabilities, for both multi and non-multispectral images.
    
    Args:
        path_image (string) - path to a .mat file containing the raw big image data.
        
    Output:
        none
    
    Note: main() function is wrapper for this function.    
    """
    
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
    
    # # 4-30-20: compare performance of ARRInet-W with ARRInet-M using just 3 channels and other channels occluded
    # im_occluded = deepcopy(im) # copy; avoid assignment by reference
    # im_occluded[:,:,3:] = MEAN_CHANNEL_PIXELVALS[3:] # occlude entire i'th channel with average value across training dataset
    
    # Tile image into 32x32 pixel patches
    list_tiles, list_fracmask_intile, list_loc = tile_data_ariraw_GSL.gettiles3d(im, segmentation, tile_size=(32,32), fracinmask=0)
    # list_tiles, list_fracmask_intile, list_loc = tile_data_ariraw_GSL.gettiles3d(im_occluded, segmentation, tile_size=(32,32), fracinmask=0)
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
    prob_trueclass_alltiles_ms = np.average(class_prob, axis=0)[trueclass_ind]
    prob_trueclass_alltiles_RGB =  np.average(class_prob_RGB, axis=0)[trueclass_ind]
    print('True class:', true_class, '(', prob_trueclass_alltiles_ms, 'MS prob,', prob_trueclass_alltiles_RGB, 'RGB prob)')
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
    print('Number of tiles with 100% tissue:', num_tiles_allmask)
    isinmask = [i==1 for i in list_fracmask_intile]
    num_tiles_correct = sum(pred_class_int[isinmask]==trueclass_ind)[0]
    frac_tilesinmask_correct = np.around(num_tiles_correct/num_tiles_allmask*100, decimals=1)
    print('Number tiles with 100% tissue correctly predicted (multispectral):', num_tiles_correct, '(' + str(frac_tilesinmask_correct) + '%)')
    num_tiles_correct_RGB = sum(pred_class_int_RGB[isinmask]==trueclass_ind)[0]
    frac_tilesinmask_correct_RGB = np.around(num_tiles_correct_RGB/num_tiles_allmask*100, decimals=1)
    print('Number tiles with 100% tissue correctly predicted (non-multispectral):', num_tiles_correct_RGB, '(' + str(frac_tilesinmask_correct_RGB) + '%)')
    prob_trueclass_inmask_ms = np.average(class_prob[:, trueclass_ind][isinmask])
    prob_trueclass_inmask_RGB =  np.average(class_prob_RGB[:, trueclass_ind][isinmask])
    print('Probability of true class (multispectral):', str(np.around(prob_trueclass_inmask_ms*100, decimals=1)) + '%')
    print('Probability of true class (non-multispectral):', str(np.around(prob_trueclass_inmask_RGB*100, decimals=1)) + '%')
    
    # Generate heat map for big picture classification
    # Load processed TIFF image for viewing
    TITLE_FONTSIZE = 20
    TEXT_SIZE = 16
    path_tiff = os.path.join(PATH_DATA, dataset, date, this_tissue)
    tiff_filename = [f for f in os.listdir(path_tiff) if 'arriwhite20' in f][0]
    tiff_img = mpimg.imread(os.path.join(path_tiff, tiff_filename))
    fig0 = plt.figure()
    imgplot = plt.imshow(tiff_img)
#    plt.colorbar()
    plt.axis('off')
    plt.title('TIFF, ' + true_class, fontsize=TITLE_FONTSIZE)
    
    # Plot heatmap of multispectral 11 class predictions
    N_COLORS = NUM_CLASSES+1 # 11 foreground, 1 background
    colormap11 = discrete_cmap(N_COLORS, base_cmap=cmap_set0gray('jet')) #get discrete colormap
    big_image_size = np.shape(tiff_img)
    
    fig1 = plt.figure()
    allclass_heatmap = tiles2im(pred_class_int+1, segmentation, big_image_size, tile_size=(32,32), fracinmask=1) # multispectral predicted class heatmap
    heatmap_plot = plt.imshow(allclass_heatmap, cmap=colormap11)
    cbar = plt.colorbar(ticks=range(N_COLORS))
    plt.clim(-0.5, N_COLORS - 0.5)
    cbar.ax.set_yticklabels(TISSUE_CLASSES, size=TEXT_SIZE)  # specify tick locations to match desired ticklabels
    plt.axis('off')
    plt.title('Multispectral prediction, ' + true_class + ' - ' + str(frac_tilesinmask_correct) + '%', fontsize=TITLE_FONTSIZE)
    
#    # Plot heatmap with seaborn
#    plt.figure()
#    sns.heatmap(allclass_heatmap[:,:,0], cmap=cmap_set0gray('Set3'), square=False,
#            cbar_kws={'label': 'Class', 'ticks': np.arange(0, N_COLORS)})
#    plt.title('Multispectral prediction, ' + true_class)
    
    # Plot heatmap of multispectral predicted true class probability
    fig2 = plt.figure()
    class_prob_trueclass = class_prob[:,trueclass_ind]
    probability_heatmap = tiles2im(class_prob_trueclass, segmentation, big_image_size, tile_size=(32,32), fracinmask=1)
    heatmap_plot2 = plt.imshow(probability_heatmap, cmap=cmap_set0gray('jet'), vmin=0, vmax=1)
    cbar2 = plt.colorbar()
    cbar2.ax.tick_params(labelsize=TEXT_SIZE)
#    cbar2.ax.yaxis.label.set_font_properties(mpl.font_manager.FontProperties(size=TEXT_SIZE))
    cbar2.set_label('Probability', size=TEXT_SIZE)
    plt.axis('off')
    plt.title('Multispectral probability, ' + true_class + ' - ' + str(np.around(prob_trueclass_inmask_ms*100, decimals=1)) + '%', fontsize=TITLE_FONTSIZE)
    
    # Plot heatmap of non-multispectral 11 class predictions
    fig3 = plt.figure()
    allclass_heatmap_RGB = tiles2im(pred_class_int_RGB+1, segmentation, big_image_size, tile_size=(32,32), fracinmask=1) # multispectral predicted class heatmap
    heatmap_plot_RGB = plt.imshow(allclass_heatmap_RGB, cmap=colormap11)
    cbar3 = plt.colorbar(ticks=range(N_COLORS))
    plt.clim(-0.5, N_COLORS - 0.5)
    cbar3.ax.set_yticklabels(TISSUE_CLASSES, size=TEXT_SIZE)  # specify tick locations to match desired ticklabels
    plt.axis('off')
    plt.title('Non-multispectral prediction, ' + true_class + ' - ' + str(frac_tilesinmask_correct_RGB) + '%', fontsize=TITLE_FONTSIZE)
    
    # Plot heatmap of non-multispectral predicted true class probability
    fig4 = plt.figure()
    class_prob_RGB_trueclass = class_prob_RGB[:, trueclass_ind]
    probability_heatmap_RGB = tiles2im(class_prob_RGB_trueclass, segmentation, big_image_size, tile_size=(32,32), fracinmask=1)
    heatmap_plot2_RGB = plt.imshow(probability_heatmap_RGB, cmap=cmap_set0gray('jet'), vmin=0, vmax=1)
    cbar4 = plt.colorbar()
    cbar4.ax.tick_params(labelsize=TEXT_SIZE)
#    cbar4.ax.yaxis.label.set_font_properties(mpl.font_manager.FontProperties(size=TEXT_SIZE))
    cbar4.set_label('Probability', size=TEXT_SIZE)
    plt.axis('off')
    plt.title('Non-multispectral probability, ' + true_class + ' - ' + str(np.around(prob_trueclass_inmask_RGB*100, decimals=1)) + '%', fontsize=TITLE_FONTSIZE)

    # Write cache of image arrinet results to pickle file
    target_folder = os.path.join(PATH_OUT, this_tissue, "")
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        
    pickle_filename_MS = 'arrinet_MS_analysis.pickle'
    pickle.dump([class_scores, class_prob, pred_class_int, pred_class_name, list_fracmask_intile, list_loc, segmentation], open(os.path.join(target_folder, pickle_filename_MS), "wb" ))
    pickle_filename_RGB = 'arrinet_RGB_analysis.pickle'
    pickle.dump([class_scores_RGB, class_prob_RGB, pred_class_int_RGB, pred_class_name_RGB, list_fracmask_intile, list_loc, segmentation], open(os.path.join(target_folder, pickle_filename_RGB), "wb" ))

    # Save figures
    # Figure saving parameters
    FIG_HEIGHT = 12 # inches
    FIG_WIDTH = 8 # inches
    FIG_DPI = 200
    
    filename_id = date + '_' + this_tissue + '_labelis_' + true_class + '_' + dataset
    fig0_filename = 'tiff_' + filename_id + '.pdf'
    fig1_filename = 'heatmap_ms_pred' + filename_id + '_acc_' + str(np.around(frac_tilesinmask_correct, decimals=0)) + '.pdf'
    fig2_filename = 'heatmap_ms_probability' + filename_id + '_acc_' + str(np.around(prob_trueclass_inmask_ms*100, decimals=0)) + '.pdf'
    fig3_filename = 'heatmap_rgb_pred' + filename_id + '_acc_' + str(np.around(frac_tilesinmask_correct_RGB, decimals=0)) + '.pdf'
    fig4_filename = 'heatmap_rgb_probability' + filename_id + '_acc_' + str(np.around(prob_trueclass_inmask_RGB*100, decimals=0)) + '.pdf'
    
    fig0.set_size_inches(FIG_HEIGHT, FIG_WIDTH)
    fig1.set_size_inches(FIG_HEIGHT, FIG_WIDTH)
    fig2.set_size_inches(FIG_HEIGHT, FIG_WIDTH)
    fig3.set_size_inches(FIG_HEIGHT, FIG_WIDTH)
    fig4.set_size_inches(FIG_HEIGHT, FIG_WIDTH)
    
    # fig0.savefig(os.path.join(target_folder, fig0_filename), bbox_inches='tight', dpi=FIG_DPI)
    # fig1.savefig(os.path.join(target_folder, fig1_filename), bbox_inches='tight', dpi=FIG_DPI)
    # fig2.savefig(os.path.join(target_folder, fig2_filename), bbox_inches='tight', dpi=FIG_DPI)
    # fig3.savefig(os.path.join(target_folder, fig3_filename), bbox_inches='tight', dpi=FIG_DPI)
    # fig4.savefig(os.path.join(target_folder, fig4_filename), bbox_inches='tight', dpi=FIG_DPI)


    # # Obtain occlusion predictions for multispectral arrinet
    # n_channels = np.shape(im)[-1]
    # cache_frac_tilesinmask_correct_occluded = []
    # cache_prob_trueclass_inmask__occluded = []
    # for i in np.arange(n_channels):
    #     im_occluded = deepcopy(im) # copy; avoid assignment by reference
    #     im_occluded[:,:,i] = MEAN_CHANNEL_PIXELVALS[i] # occlude entire i'th channel with average value across training dataset
        
    #     # Obtain only tiles 100% in mask
    #     list_tiles_occluded, _, _ = tile_data_ariraw_GSL.gettiles3d(im_occluded, segmentation, tile_size=(32,32), fracinmask=1)
    #     stack_tiles_occluded = np.stack(list_tiles_occluded, axis=0) # 4-D ndarray of shape (N=1980, H=32, W=32, C=21)
    #     stack_tiles_occluded = np.transpose(stack_tiles_occluded, axes=(0, 3, 1, 2)) # permute dimensions from ndarray (N, H, W, C) to (N, C, H, W) 
    #     class_scores_occluded, class_prob_occluded, pred_class_int_occluded, pred_class_name_occluded = arrinet_classify.classify(stack_tiles_occluded, isprocessed=False, ismultispectral=True)

    #     # cache in-mask % tiles predicted as true class and average probability of true class
    #     frac_tiles_correct_occluded = sum(pred_class_int_occluded==trueclass_ind)[0]/num_tiles_allmask
    #     prob_trueclass_ave_occluded = np.average(class_prob_occluded[:,trueclass_ind])
    #     cache_frac_tilesinmask_correct_occluded.append(frac_tiles_correct_occluded)
    #     cache_prob_trueclass_inmask_occluded.append(prob_trueclass_ave_occluded)
        
    #     print('Occluding channel', str(i), 'results in', frac_tiles_correct_occluded, 'tiles correct and ', prob_trueclass_ave_occluded, 'average probability of true class.')

    # # Write cache of occluded image arrinet results to pickle file
    # pickle_filename = 'occlusion.pickle'
    # pickle.dump([cache_frac_tilesinmask_correct_occluded, cache_prob_trueclass_inmask_occluded], open(os.path.join(target_folder, pickle_filename), "wb" ))

def analyze_occlusions():
    """Analyze pickle files of occlusion outputs"""
    dirs = [f for f in os.listdir(PATH_EVAL_OUTPUT) if os.path.isdir(os.path.join(PATH_EVAL_OUTPUT, f))]
    df1_pred = pd.DataFrame()
    df2_prob = pd.DataFrame()
    for this_dir in dirs:
        my_tissue = this_dir
        this_dir = os.path.join(PATH_EVAL_OUTPUT, this_dir)
        pickle_path = [f for f in os.listdir(this_dir) if f.endswith('.pickle')]
        [cache_frac_tilesinmask_correct_occluded, cache_prob_trueclass_inmask_occluded] = pickle.load(open(os.path.join(this_dir, pickle_path[0]), 'rb'))
        print('Occluding tissue ', my_tissue, ':', cache_frac_tilesinmask_correct_occluded, '\nprobability', cache_prob_trueclass_inmask_occluded)

        # Add row to dataframe
        new_pred = pd.DataFrame({my_tissue: cache_frac_tilesinmask_correct_occluded})
        new_prob = pd.DataFrame({my_tissue: cache_prob_trueclass_inmask_occluded})
        df1_pred = pd.concat([df1_pred, new_pred.T], ignore_index=True)
        df2_prob = pd.concat([df2_prob, new_prob.T], ignore_index=True)
        
    # Write compiled outputs of occlusion results to csv file
    pred_csv_filename = 'occlusions_pred.csv'
    prob_csv_filename = 'occlusions_prob.csv'
    df1_pred.to_csv(os.path.join(PATH_EVAL_OUTPUT, pred_csv_filename))
    df2_prob.to_csv(os.path.join(PATH_EVAL_OUTPUT, prob_csv_filename))

def plot_occlusions(path_csv):
    """Plot CSV files of occlusion differences in prediction accuracy
    Args:
        path_csv (str) - path to CSV file containing differences of values. Values in B2:V13 of CSV file
    """
    this_data = pd.read_csv(path_csv)
    differences = this_data.iloc[:,1:]
    # differences = differences.to_numpy()
    print(differences.shape)
    
    class_labels = TISSUE_CLASSES[1:]
    class_labels.insert(-2,'PerichondriumWCartilage') # Add back in label for perichondrium to match # rows
    n_channels = len(MEAN_CHANNEL_PIXELVALS)
    channel_blocked = np.arange(n_channels)

    sns.set(font_scale=1.4)    
    fig0 = plt.figure(figsize=(20,10))
    sns.heatmap(differences, cmap='RdBu_r', vmin=-1, vmax=1, center=0, square=True,
            xticklabels=channel_blocked, yticklabels=class_labels,
            cbar_kws={'label': 'Change'})
    fig0.savefig(os.path.join(PATH_EVAL_OUTPUT,'occlusions_plot1.pdf'), dpi=300)
    
    fig1 = plt.figure(figsize=(20,10)) # Figure with values printed on each tile
    sns.heatmap(differences, cmap='RdBu_r', vmin=-1, vmax=1, center=0, square=True, annot=np.around(differences.to_numpy(), decimals=2),
            xticklabels=channel_blocked, yticklabels=class_labels,
            cbar_kws={'label': 'Change'})
    fig1.savefig(os.path.join(PATH_EVAL_OUTPUT,'occlusions_plot2.pdf'), dpi=300)

def compare_parotid_nerve(path_folder): 
    """Analyze parotid versus nerve prediction accuracy and probability heatmaps
    Run evaluate_bigimage first on parotid and nerve images first
    
    Args:
        path_folder (str) - path to output folder containing tissue output directories with pickle files of parotid and nerve classification values
    """
    dir_output_parotid = os.path.join(path_folder, 'Parotid/')
    dir_output_nerve = os.path.join(path_folder, 'Nerve/')
    
    INDEX_PAROTID = 8
    INDEX_NERVE = 7
    
    # Multispectral parotid image analysis
    pickle_parotid = [f for f in os.listdir(dir_output_parotid) if f.endswith('analysis.pickle')]
    [class_scores, class_prob, pred_class_int, pred_class_name, list_fracmask_intile, list_loc, segmentation_parotid] = pickle.load(open(os.path.join(dir_output_parotid, pickle_parotid[0]), 'rb'))
    isinmask = [i==1 for i in list_fracmask_intile]
    parotid_prob_parotid = class_prob[:, INDEX_PAROTID]
    parotid_prob_nerve = class_prob[:, INDEX_NERVE]
    # adjust probabilities to account for only 2 options: parotid and nerve
    parotid_prob_parotid = parotid_prob_parotid/(parotid_prob_parotid + parotid_prob_nerve)
    ave_prob_parotid = np.average(parotid_prob_parotid[isinmask])
    num_tilesinmask_parotid = sum(isinmask)
    parotid_frac_tilesinmask_correct = sum(parotid_prob_parotid[isinmask] > 0.5)/num_tilesinmask_parotid

    # Plot heatmap of multispectral parotid image probability of parotid
    FIG_HEIGHT = 12 # inches
    FIG_WIDTH = 8 # inches
    TITLE_FONTSIZE = 20
    TEXT_SIZE = 16
    fig0 = plt.figure(figsize=(FIG_HEIGHT, FIG_WIDTH))
    probability_heatmap = tiles2im(parotid_prob_parotid, segmentation_parotid, np.shape(segmentation_parotid)[:2], tile_size=(32,32), fracinmask=1)
    heatmap_plot0 = plt.imshow(probability_heatmap, cmap=cmap_set0gray('jet'), vmin=0, vmax=1)
    cbar0 = plt.colorbar()
    cbar0.ax.tick_params(labelsize=TEXT_SIZE)
    cbar0.set_label('Probability', size=TEXT_SIZE)
    plt.axis('off')
    plt.title('Multispectral parotid, probability - ' + str(np.around(ave_prob_parotid*100, decimals=1)) + '%, prediction ' + str(np.around(parotid_frac_tilesinmask_correct*100, decimals=1)) + '%', fontsize=TITLE_FONTSIZE)

    # Plot heatmap of non-multispectral parotid image probability of parotid
    [class_scores_RGB, class_prob_RGB, pred_class_int_RGB, pred_class_name_RGB, list_fracmask_intile, list_loc, segmentation_parotid] = pickle.load(open(os.path.join(dir_output_parotid, pickle_parotid[1]), 'rb'))
    parotid_prob_parotid_RGB = class_prob_RGB[:, INDEX_PAROTID]
    parotid_prob_nerve_RGB = class_prob_RGB[:, INDEX_NERVE]
    parotid_prob_parotid_RGB = parotid_prob_parotid_RGB/(parotid_prob_parotid_RGB + parotid_prob_nerve_RGB) # adjust probabilities to account for only 2 options: parotid and nerve
    ave_prob_parotid_RGB = np.average(parotid_prob_parotid_RGB[isinmask])
    parotid_frac_tilesinmask_correct_RGB = sum(parotid_prob_parotid_RGB[isinmask] > 0.5)/num_tilesinmask_parotid
    
    fig1 = plt.figure(figsize=(FIG_HEIGHT, FIG_WIDTH))
    probability_heatmap1 = tiles2im(parotid_prob_parotid_RGB, segmentation_parotid, np.shape(segmentation_parotid)[:2], tile_size=(32,32), fracinmask=1)
    heatmap_plot1 = plt.imshow(probability_heatmap1, cmap=cmap_set0gray('jet'), vmin=0, vmax=1)
    cbar1 = plt.colorbar()
    cbar1.ax.tick_params(labelsize=TEXT_SIZE)
    cbar1.set_label('Probability', size=TEXT_SIZE)
    plt.axis('off')
    plt.title('Non-multispectral parotid, probability - ' + str(np.around(ave_prob_parotid_RGB*100, decimals=1)) + '%, prediction ' + str(np.around(parotid_frac_tilesinmask_correct_RGB*100, decimals=1)) + '%', fontsize=TITLE_FONTSIZE)


    # Multispectral nerve image analysis
    pickle_nerve = [f for f in os.listdir(dir_output_nerve) if f.endswith('analysis.pickle')]
    [class_scores, class_prob, pred_class_int, pred_class_name, list_fracmask_intile, list_loc, segmentation_nerve] = pickle.load(open(os.path.join(dir_output_nerve, pickle_nerve[0]), 'rb'))
    isinmask = [i==1 for i in list_fracmask_intile]
    nerve_prob_parotid = class_prob[:, INDEX_PAROTID]
    nerve_prob_nerve = class_prob[:, INDEX_NERVE]
    # adjust probabilities to account for only 2 options: parotid and nerve
    nerve_prob_nerve = nerve_prob_nerve/(nerve_prob_nerve + nerve_prob_parotid)
    ave_prob_nerve = np.average(nerve_prob_nerve[isinmask])
    num_tilesinmask_nerve = sum(isinmask)
    nerve_frac_tilesinmask_correct = sum(nerve_prob_nerve[isinmask] > 0.5)/num_tilesinmask_nerve

    # Plot heatmap of multispectral nerve image probability of nerve
    fig2 = plt.figure(figsize=(FIG_HEIGHT, FIG_WIDTH))
    probability_heatmap2 = tiles2im(nerve_prob_nerve, segmentation_nerve, np.shape(segmentation_nerve)[:2], tile_size=(32,32), fracinmask=1)
    heatmap_plot2 = plt.imshow(probability_heatmap2, cmap=cmap_set0gray('jet'), vmin=0, vmax=1)
    cbar2 = plt.colorbar()
    cbar2.ax.tick_params(labelsize=TEXT_SIZE)
    cbar2.set_label('Probability', size=TEXT_SIZE)
    plt.axis('off')
    plt.title('Multispectral nerve, probability - ' + str(np.around(ave_prob_nerve*100, decimals=1)) + '%, prediction ' + str(np.around(nerve_frac_tilesinmask_correct*100, decimals=1)) + '%', fontsize=TITLE_FONTSIZE)

    # Plot heatmap of non-multispectral nerve image probability of parotid
    [class_scores_RGB, class_prob_RGB, pred_class_int_RGB, pred_class_name_RGB, list_fracmask_intile, list_loc, segmentation_nerve] = pickle.load(open(os.path.join(dir_output_nerve, pickle_nerve[1]), 'rb'))
    nerve_prob_parotid_RGB = class_prob_RGB[:, INDEX_PAROTID]
    nerve_prob_nerve_RGB = class_prob_RGB[:, INDEX_NERVE]
    # adjust probabilities to account for only 2 options: parotid and nerve
    nerve_prob_nerve_RGB = nerve_prob_nerve_RGB/(nerve_prob_nerve_RGB + nerve_prob_parotid_RGB)
    ave_prob_nerve_RGB = np.average(nerve_prob_nerve_RGB[isinmask])
    nerve_frac_tilesinmask_correct_RGB = sum(nerve_prob_nerve_RGB[isinmask] > 0.5)/num_tilesinmask_nerve
    
    fig3 = plt.figure(figsize=(FIG_HEIGHT, FIG_WIDTH))
    probability_heatmap3 = tiles2im(nerve_prob_nerve_RGB, segmentation_nerve, np.shape(segmentation_nerve)[:2], tile_size=(32,32), fracinmask=1)
    heatmap_plot3 = plt.imshow(probability_heatmap3, cmap=cmap_set0gray('jet'), vmin=0, vmax=1)
    cbar3 = plt.colorbar()
    cbar3.ax.tick_params(labelsize=TEXT_SIZE)
    cbar3.set_label('Probability', size=TEXT_SIZE)
    plt.axis('off')
    plt.title('Non-multispectral nerve, probability - ' + str(np.around(ave_prob_nerve_RGB*100, decimals=1)) + '%, prediction ' + str(np.around(nerve_frac_tilesinmask_correct_RGB*100, decimals=1)) + '%', fontsize=TITLE_FONTSIZE)

    # Save figures
    FIG_DPI = 200
    
    fig0_filename = 'heatmap_ms_parotid_probability_parotid_vs_nerve_acc_' + str(np.around(ave_prob_parotid*100, decimals=0)) + '.pdf'
    fig1_filename = 'heatmap_rgb_parotid_probability_parotid_vs_nerve_acc_' + str(np.around(ave_prob_parotid_RGB*100, decimals=0)) + '.pdf'
    fig2_filename = 'heatmap_ms_nerve_probability_nerve_vs_parotid_acc_' + str(np.around(ave_prob_nerve*100, decimals=0)) + '.pdf'
    fig3_filename = 'heatmap_rgb_nerve_probability_nerve_vs_parotid_acc_' + str(np.around(ave_prob_nerve_RGB*100, decimals=0)) + '.pdf'
    
    fig0.savefig(os.path.join(path_folder, fig0_filename), bbox_inches='tight', dpi=FIG_DPI)
    fig1.savefig(os.path.join(path_folder, fig1_filename), bbox_inches='tight', dpi=FIG_DPI)
    fig2.savefig(os.path.join(path_folder, fig2_filename), bbox_inches='tight', dpi=FIG_DPI)
    fig3.savefig(os.path.join(path_folder, fig3_filename), bbox_inches='tight', dpi=FIG_DPI)
    
    
    ## occlusion analysis
    # Obtain occlusion predictions for multispectral arrinet - parotid
    path_image = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/arriraw_data/20190520/20190520-Parotid_GSL.mat'
    im = tile_data_ariraw_GSL.import_ariraw_matlab(path_image) # 21 channel multispectral image, float64 ndarray shape (H, W, C) 
    n_channels = np.shape(im)[-1]
    # cache_frac_tilesinmask_correct_occluded = []
    # cache_prob_trueclass_inmask_occluded = []
    parotid_occlusions_pred = pd.DataFrame()
    parotid_occlusions_prob = pd.DataFrame()
    for i in np.arange(n_channels):
        im_occluded = deepcopy(im) # copy; avoid assignment by reference
        im_occluded[:,:,i] = MEAN_CHANNEL_PIXELVALS[i] # occlude entire i'th channel with average value across training dataset
        
        # Obtain only tiles 100% in mask
        list_tiles_occluded, _, _ = tile_data_ariraw_GSL.gettiles3d(im_occluded, segmentation_parotid, tile_size=(32,32), fracinmask=1)
        stack_tiles_occluded = np.stack(list_tiles_occluded, axis=0) # 4-D ndarray of shape (N=1980, H=32, W=32, C=21)
        stack_tiles_occluded = np.transpose(stack_tiles_occluded, axes=(0, 3, 1, 2)) # permute dimensions from ndarray (N, H, W, C) to (N, C, H, W) 
        class_scores_occluded, class_prob_occluded, pred_class_int_occluded, pred_class_name_occluded = arrinet_classify.classify(stack_tiles_occluded, isprocessed=False, ismultispectral=True)

        # cache in-mask % tiles predicted as true class and average probability of true class
        class_prob_occluded_parotid = class_prob_occluded[:, INDEX_PAROTID]
        class_prob_occluded_nerve = class_prob_occluded[:, INDEX_NERVE]
        class_prob_occluded_parotid = class_prob_occluded_parotid/(class_prob_occluded_parotid + class_prob_occluded_nerve)
        frac_tiles_correct_occluded = sum(class_prob_occluded_parotid > 0.5)/num_tilesinmask_parotid
        prob_trueclass_ave_occluded = np.average(class_prob_occluded_parotid)
        # cache_frac_tilesinmask_correct_occluded.append(frac_tiles_correct_occluded)
        # cache_prob_trueclass_inmask_occluded.append(prob_trueclass_ave_occluded)
        parotid_occlusions_pred[i] = [frac_tiles_correct_occluded]
        parotid_occlusions_prob[i] = [prob_trueclass_ave_occluded]
        
        print('Occluding parotid channel', str(i), 'results in', frac_tiles_correct_occluded, 'tiles correct and ', prob_trueclass_ave_occluded, 'average probability of true class.')
        
    # cache_frac_tilesinmask_correct_occluded_parotid = deepcopy(cache_frac_tilesinmask_correct_occluded)
    # cache_prob_trueclass_inmask_occluded_parotid = deepcopy(cache_prob_trueclass_inmask_occluded)
    
    # Obtain occlusion predictions for multispectral arrinet - nerve
    path_image = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/arriraw_data/20190520/20190520-Nerve_GSL.mat'
    im = tile_data_ariraw_GSL.import_ariraw_matlab(path_image) # 21 channel multispectral image, float64 ndarray shape (H, W, C) 
    n_channels = np.shape(im)[-1]
    # cache_frac_tilesinmask_correct_occluded = []
    # cache_prob_trueclass_inmask_occluded = []
    nerve_occlusions_pred = pd.DataFrame()
    nerve_occlusions_prob = pd.DataFrame()
    for i in np.arange(n_channels):
        im_occluded = deepcopy(im) # copy; avoid assignment by reference
        im_occluded[:,:,i] = MEAN_CHANNEL_PIXELVALS[i] # occlude entire i'th channel with average value across training dataset
        
        # Obtain only tiles 100% in mask
        list_tiles_occluded, _, _ = tile_data_ariraw_GSL.gettiles3d(im_occluded, segmentation_nerve, tile_size=(32,32), fracinmask=1)
        stack_tiles_occluded = np.stack(list_tiles_occluded, axis=0) # 4-D ndarray of shape (N=1980, H=32, W=32, C=21)
        stack_tiles_occluded = np.transpose(stack_tiles_occluded, axes=(0, 3, 1, 2)) # permute dimensions from ndarray (N, H, W, C) to (N, C, H, W) 
        class_scores_occluded, class_prob_occluded, pred_class_int_occluded, pred_class_name_occluded = arrinet_classify.classify(stack_tiles_occluded, isprocessed=False, ismultispectral=True)

        # cache in-mask % tiles predicted as true class and average probability of true class
        class_prob_occluded_parotid = class_prob_occluded[:, INDEX_PAROTID]
        class_prob_occluded_nerve = class_prob_occluded[:, INDEX_NERVE]
        class_prob_occluded_nerve = class_prob_occluded_nerve/(class_prob_occluded_parotid + class_prob_occluded_nerve)
        frac_tiles_correct_occluded = sum(class_prob_occluded_nerve > 0.5)/num_tilesinmask_nerve
        prob_trueclass_ave_occluded = np.average(class_prob_occluded_nerve)
        # cache_frac_tilesinmask_correct_occluded.append(frac_tiles_correct_occluded)
        # cache_prob_trueclass_inmask_occluded.append(prob_trueclass_ave_occluded)
        nerve_occlusions_pred[i] = [frac_tiles_correct_occluded]
        nerve_occlusions_prob[i] = [prob_trueclass_ave_occluded]
        
        print('Occluding nerve channel', str(i), 'results in', frac_tiles_correct_occluded, 'tiles correct and ', prob_trueclass_ave_occluded, 'average probability of true class.')
        
    # cache_frac_tilesinmask_correct_occluded_nerve = deepcopy(cache_frac_tilesinmask_correct_occluded)
    # cache_prob_trueclass_inmask_occluded_nerve = deepcopy(cache_prob_trueclass_inmask_occluded)

    # Analyze occlusion of parotid vs nerve
    diff_parotid_occlusions_pred = parotid_occlusions_pred - parotid_frac_tilesinmask_correct
    diff_parotid_occlusions_prob = parotid_occlusions_prob - ave_prob_parotid
    diff_nerve_occlusions_pred = nerve_occlusions_pred - nerve_frac_tilesinmask_correct
    diff_nerve_occlusions_prob = nerve_occlusions_prob - ave_prob_nerve
    
    BINARY_TISSUE_CLASSES = ['Parotid', 'Nerve']
    diff_parotid_nerve_occlusions_pred = pd.concat([diff_parotid_occlusions_pred, diff_nerve_occlusions_pred])
    diff_parotid_nerve_occlusions_pred.index = BINARY_TISSUE_CLASSES
    diff_parotid_nerve_occlusions_prob = pd.concat([diff_parotid_occlusions_prob, diff_nerve_occlusions_prob])
    diff_parotid_nerve_occlusions_prob.index = BINARY_TISSUE_CLASSES
    
   
    # Plot occlusion analysis of parotid vs nerve
    print('diff_parotid_nerve_occlusions_pred shape:', diff_parotid_nerve_occlusions_pred.shape)
    print('diff_parotid_nerve_occlusions_prob shape:', diff_parotid_nerve_occlusions_prob.shape)
    
    class_labels = BINARY_TISSUE_CLASSES
    n_channels = len(MEAN_CHANNEL_PIXELVALS)
    channel_blocked = np.arange(n_channels)

    sns.set(font_scale=1.4)    
    fig4 = plt.figure(figsize=(20,10))
    sns.heatmap(diff_parotid_nerve_occlusions_pred, cmap='RdBu_r', vmin=-1, vmax=1, center=0, square=True,
            xticklabels=False, yticklabels=class_labels,
            cbar_kws={'label': 'Change'})
    fig4.savefig(os.path.join(PATH_EVAL_OUTPUT,'occlusions_parotid_nerve_pred.pdf'), dpi=300)
    
    fig5 = plt.figure(figsize=(20,10)) # Figure with values printed on each tile
    sns.heatmap(diff_parotid_nerve_occlusions_prob, cmap='RdBu_r', vmin=-1, vmax=1, center=0, square=True,
            xticklabels=False, yticklabels=class_labels,
            cbar_kws={'label': 'Change'})
    fig5.savefig(os.path.join(PATH_EVAL_OUTPUT,'occlusions_parotid_nerve_prob.pdf'), dpi=300)
    
    # Also save occlusion plots with annotations to be able to extracte quantitative values later
    fig6 = plt.figure(figsize=(20,10))
    sns.heatmap(diff_parotid_nerve_occlusions_pred, cmap='RdBu_r', vmin=-1, vmax=1, center=0, square=True, annot=np.around(diff_parotid_nerve_occlusions_pred.to_numpy(), decimals=2),
            xticklabels=False, yticklabels=class_labels,
            cbar_kws={'label': 'Change'})
    fig6.savefig(os.path.join(PATH_EVAL_OUTPUT,'occlusions_parotid_nerve_pred_annotated.pdf'), dpi=300)
    
    fig7 = plt.figure(figsize=(20,10)) # Figure with values printed on each tile
    sns.heatmap(diff_parotid_nerve_occlusions_prob, cmap='RdBu_r', vmin=-1, vmax=1, center=0, square=True, annot=np.around(diff_parotid_nerve_occlusions_prob.to_numpy(), decimals=2),
            xticklabels=False, yticklabels=class_labels,
            cbar_kws={'label': 'Change'})
    fig7.savefig(os.path.join(PATH_EVAL_OUTPUT,'occlusions_parotid_nerve_prob_annotated.pdf'), dpi=300)

def plot_histogram_rawimage(path_image):
    """Analyze histograms of each channel to double-check exposure durations of ARRIwhite and white-mix lights are same
    Requested by Dr. Joyce Farrell 4-16-20
    
    Args:
        path_image (str) - path to .mat file containing raw tissue multispectral image
        
    TODO: this code function is incomplete as of 4-30-2020
    """
    # Read in image to python workspace
    im = tile_data_ariraw_GSL.import_ariraw_matlab(path_image) # 21 channel multispectral image, float64 ndarray shape (H, W, C) 
    
    # Calculate histogram of each channel
    num_bins = 100
    num_channels = np.shape(im)[2]
    cache_histograms = []
    # for i in np.arange(num_channels):
    for i in np.arange(num_channels):
        print('Working on', i, 'out of', num_channels)
        # hist, bin_edges = np.histogram(im[:,:,i], bins=num_bins)
        plt.figure()
        n, bins, patches = plt.hist(im[:,:,i].flatten(), bins=num_bins, facecolor='blue', alpha=1, histtype='step', density=True)  # arguments are passed to np.histogram
        plt.title("Histogram of pixels, channel " + str(i) + ", bins =" + str(num_bins))
        plt.xlabel('Bin')
        plt.ylabel('Probability')
        
        # cache_histograms.append(hist)

    plt.show()        
    print('done')

def roc_arrinet(path_folder):
    """Generate ROC curves and evaluate AUC of Arrinet classification of directory of whole raw images (eg test dataset).
    Compare performance of multispectral and non-multispectral data.
    
    Args:
        path_folder (string) - path to a directory containing .mat files of raw big image data.
        
    Output:
        none
    """
    
    # Initialize true and predicted label vectors
    n_classes = len(TISSUE_CLASSES_FINAL)
    y_true = np.array([], dtype=np.int64).reshape(0,n_classes) 
    y_score_ms = np.array([], dtype=np.int64).reshape(0,n_classes) 
    y_score_rgb = np.array([], dtype=np.int64).reshape(0,n_classes) 
    
    # Read in raw images in folder path
    only_matfiles = [f for f in os.listdir(path_folder) if f.endswith(".mat")]
    num_files = len(only_matfiles)
    for i, file in enumerate(only_matfiles):
        print('\n Working on image', str(i), 'out of', str(num_files))
        path_image = os.path.join(path_folder, file)
    
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

        # Create vector of binarized true class label
        num_tiles = class_scores.shape[0]
        true_class = np.matlib.repmat(true_class, num_tiles, 1)
        this_label = sklearn.preprocessing.label_binarize(true_class, classes=TISSUE_CLASSES_FINAL)
        print(np.shape(this_label))
        print(np.shape(y_true))

        # Update true and predicted label vectors
        y_true = np.concatenate((y_true, this_label), axis=0) 
        y_score_ms = np.concatenate((y_score_ms, class_scores), axis=0) 
        y_score_rgb = np.concatenate((y_score_rgb, class_scores_RGB), axis=0) 
        
    print('Analyzing ROC....')
    # Compute ROC curve and ROC area for each class
    y_test = y_true
    for y_score in [y_score_ms, y_score_rgb]:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])
        
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])
        
        # Plot all ROC curves
        fig0 = plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)
        
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)
        
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        lw = 2
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
        
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
            # Save figures
    
        plt.show()
        
        # Figure saving parameters
        FIG_HEIGHT = 12 # inches
        FIG_WIDTH = 8 # inches
        fig0.set_size_inches(FIG_HEIGHT, FIG_WIDTH)
    
    # TODO: Calculate significant difference statistic of AUC ARRInet-M vs ARRInet-W

def main():
    # #User select a whole raw image (.mat)
    # print('Select an unprocessed, multispectral whole image (.mat):')
    # path_image = mat.uigetfile(initialdir='C:/Users/CTLab/Documents/George/Python_data/arritissue_data/', title='Select file', filetypes=(("MAT-file","*.mat"),("all files","*.*")))
    # print(path_image)
    # # Evaluate on one, user-selected big raw image
    # evaluate_bigimage(path_image)
    
    # # Alternatively, evaluate on all tissues in one acquisition date (folder)
    # # Wrapper function for evaluate_bigimage()
    # print('Select a folder containing unprocessed, multispectral whole images (.mat files). Folder name should be acquisition date:')
    # path_dir = mat.uigetdir(initialdir='C:/Users/CTLab/Documents/George/Python_data/arritissue_data/', title='Select folder')
    # print(path_dir)
    # only_matfiles = [f for f in os.listdir(path_dir) if f.endswith(".mat")]
    # num_files = len(only_matfiles)
    # for i, file in enumerate(only_matfiles):
    #     print('\n Working on image', str(i), 'out of', str(num_files))
    #     path_image = os.path.join(path_dir, file)
    #     evaluate_bigimage(path_image)
        
    # # Aggregate occlusion output pickle files into CSV spreadsheet with values across all classes
    # analyze_occlusions()
    
    # # Plot occlusion analysis change in predictions/probabilities as heatmap
    # # First manually update CSV files to reflect changes of occluded predictions/probabilities from pre-occlusion values
    # path_csv = mat.uigetfile(initialdir=PATH_EVAL_OUTPUT, filetypes=(("CSV files", "*.csv"), ("all files", "*.*")))
    # plot_occlusions(path_csv)
    
    # # Analyze parotid vs nerve binary classification  -  predicted and probability heatmaps
    # path_outputdir = mat.uigetdir(initialdir=PATH_EVAL_OUTPUT, title='Select directory of output analysis')
    # print(path_outputdir)
    # compare_parotid_nerve(path_outputdir)
    
    # # 4-17-20: Plot histogram of channels for user-selected raw multispectral image
    # print('Select an unprocessed, multispectral whole image (.mat):')
    # path_image = mat.uigetfile(initialdir='C:/Users/CTLab/Documents/George/Python_data/arritissue_data/', title='Select file', filetypes=(("MAT-file","*.mat"),("all files","*.*")))
    # print(path_image)
    # plot_histogram_rawimage(path_image)
    
    # 4-30-20: Calculate ROC and AUC of ARRInet classifiers to compare performance
    print('Select a folder containing unprocessed, multispectral whole images (.mat files). Folder name should be acquisition date:')
    path_dir = mat.uigetdir(initialdir='C:/Users/CTLab/Documents/George/Python_data/arritissue_data/', title='Select folder')
    print(path_dir)
    roc_arrinet(path_dir)
    
    print('Done.')
    
if __name__=='__main__':
    main()
