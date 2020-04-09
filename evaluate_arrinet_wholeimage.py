# -*- coding: utf-8 -*-
"""
evaluate_arrinet_wholeimage.py

Evaluate ARRInet classification performance on whole tissue image

Created on Thu Apr  2 11:28:38 2020

@author: George Liu
Last edit: 4-7-2020

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

    # # Save figures
    # # Figure saving parameters
    # FIG_HEIGHT = 12 # inches
    # FIG_WIDTH = 8 # inches
    # FIG_DPI = 200
    
    # filename_id = date + '_' + this_tissue + '_labelis_' + true_class + '_' + dataset
    # fig0_filename = 'tiff_' + filename_id + '.pdf'
    # fig1_filename = 'heatmap_ms_pred' + filename_id + '_acc_' + str(np.around(frac_tilesinmask_correct, decimals=0)) + '.pdf'
    # fig2_filename = 'heatmap_ms_probability' + filename_id + '_acc_' + str(np.around(prob_trueclass_inmask_ms*100, decimals=0)) + '.pdf'
    # fig3_filename = 'heatmap_rgb_pred' + filename_id + '_acc_' + str(np.around(frac_tilesinmask_correct_RGB, decimals=0)) + '.pdf'
    # fig4_filename = 'heatmap_rgb_probability' + filename_id + '_acc_' + str(np.around(prob_trueclass_inmask_RGB*100, decimals=0)) + '.pdf'
    
    # target_folder = os.path.join(PATH_OUT, this_tissue, "")
    # if not os.path.exists(target_folder):
    #     os.makedirs(target_folder)
    
    # fig0.set_size_inches(FIG_HEIGHT, FIG_WIDTH)
    # fig1.set_size_inches(FIG_HEIGHT, FIG_WIDTH)
    # fig2.set_size_inches(FIG_HEIGHT, FIG_WIDTH)
    # fig3.set_size_inches(FIG_HEIGHT, FIG_WIDTH)
    # fig4.set_size_inches(FIG_HEIGHT, FIG_WIDTH)
    
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
    #     cache_prob_trueclass_inmask__occluded.append(prob_trueclass_ave_occluded)
        
    #     print('Occluding channel', str(i), 'results in', frac_tiles_correct_occluded, 'tiles correct and ', prob_trueclass_ave_occluded, 'average probability of true class.')

    # # Write cache of occluded image arrinet results to pickle file
    # pickle_filename = 'occlusion.pickle'
    # pickle.dump([cache_frac_tilesinmask_correct_occluded, cache_prob_trueclass_inmask__occluded], open(os.path.join(target_folder, pickle_filename), "wb" ))

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
        name_metric (str) - name of metric, e.g. percent of correctly predicted tiles or probability of true class
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

def main():
    # #User select a whole raw image (.mat)
    # print('Select an unprocessed, multispectral whole image (.mat):')
    # path_image = mat.uigetfile(initialdir='C:/Users/CTLab/Documents/George/Python_data/arritissue_data/', title='Select file', filetypes=(("MAT-file","*.mat"),("all files","*.*")))
    # print(path_image)
    # # Evaluate on one, user-selected big raw image
    # evaluate_bigimage(path_image)
    
    # Alternatively, evaluate on all tissues in one acquisition date (folder)
    # Wrapper function for evaluate_bigimage()
    print('Select a folder containing unprocessed, multispectral whole images (.mat files). Folder name should be acquisition date:')
    path_dir = mat.uigetdir(initialdir='C:/Users/CTLab/Documents/George/Python_data/arritissue_data/', title='Select folder')
    print(path_dir)
    only_matfiles = [f for f in os.listdir(path_dir) if f.endswith(".mat")]
    num_files = len(only_matfiles)
    for i, file in enumerate(only_matfiles):
        print('\n Working on image', str(i), 'out of', str(num_files))
        path_image = os.path.join(path_dir, file)
        evaluate_bigimage(path_image)
        
    # # Aggregate occlusion output pickle files into CSV spreadsheet with values across all classes
    # analyze_occlusions()
    
    # # Plot occlusion analysis change in predictions/probabilities as heatmap
    # # First manually update CSV files to reflect changes of occluded predictions/probabilities from pre-occlusion values
    # path_csv = mat.uigetfile(initialdir=PATH_EVAL_OUTPUT, filetypes=(("CSV files", "*.csv"), ("all files", "*.*")))
    # plot_occlusions(path_csv)
    
    print('Done.')
    
if __name__=='__main__':
    main()
