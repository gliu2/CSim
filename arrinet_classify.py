# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:18:04 2020

Classify multispectral 21-channel image patches (32x32)


@author: George Liu
Last edit: 4-7-2020

Dependencies: mat.py, densenet_av.py, plot_confusion_matrix.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import mat
import pickle
import densenet_av

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

STD_CHANNEL_PIXELVALS = np.array([
457.949,
748.95,
471.184,
1.27631,
1.25238,
7.24421,
2.57426,
14.4641,
3.2394,
15.3003,
9.14414,
9.35083,
7.06956,
0.877986,
0.962064,
0.857779,
0.661015,
0.911388,
21.148,
22.4566,
8.2226
])

NUM_CLASSES = 11 # cartilage and perichondrium are joined into 1 class
NUM_CHANNELS_MS = 21 # number of multispectral image channels
NUM_CHANNELS_RGB = 3 # number of non-multispectral image channels
MEAN_CHANNEL_PIXELVALS = np.reshape(MEAN_CHANNEL_PIXELVALS, (1, NUM_CHANNELS_MS, 1, 1))
STD_CHANNEL_PIXELVALS = np.reshape(STD_CHANNEL_PIXELVALS, (1, NUM_CHANNELS_MS, 1, 1))

TISSUE_CLASSES = ["Artery",
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

# Hard-coded paths to Arrinet weights files on CTlab computer
UNPROCESSED_MS_MODEL_PATH = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/arrinet_trained_models/unprocessed_images_models/multispectral/modelparam_20200130-133426_multispecTrue_nclass11_pretrainFalse_batch128_epoch10_lr0.001_L2reg0.001_DROPOUT0.0_val0.6823.pt'
UNPROCESSED_RGB_MODEL_PATH = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/arrinet_trained_models/unprocessed_images_models/rgb/modelparam_20200130-133426_multispecFalse_nclass11_pretrainFalse_batch128_epoch10_lr0.001_L2reg0.001_DROPOUT0.0_val0.6117.pt'
PROCESSED_RGB_MODEL_PATH = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/arrinet_trained_models/processed_images_models/rgb/modelparam_20190821-034714_multispecFalse_nclass11_pretrainFalse_batch128_epoch50_lr0.00013998960149928794_L2reg0.00444357531371092_DROPOUT0.0009256145730315746_val0.6389.pt'
PROCESSED_MS_MODEL_PATH = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/arrinet_trained_models/processed_images_models/multispectral/modelparam_20190822-140728_multispecTrue_nclass11_pretrainFalse_batch128_epoch40_lr0.00010188827580231053_L2reg0.005257047354276449_DROPOUT0.021436853591435358_val0.6773.pt'

def class_str2int(class_name):
    """Convert class name (string) to categorical index
    
    Args:
        class_name (string) - name of tissue class
        
    Output:
        class_int (int) - categorical index of class name
    
    TODO: make case insensitive
    """
    class_int = TISSUE_CLASSES.index(class_name)
    return class_int

def pickle_loader(input_file):
    item = pickle.load(open(input_file, 'rb'))
    return item

def normalize_image(x):
    """Normalize image patch(es) by training dataset per-channel mean and std, prior to classification.
    #
    # Input:
    #    x - multispectral image patch(es), ndarray matrix of shape (N, C, H, W). Also works if shape is (C, H, W).
    """
    if np.ndim(x) == 3:
        n_channels = np.shape(x)[0]
    elif np.ndim(x) == 4:
        n_channels = np.shape(x)[1]
    else: print('ERROR: input image dimensions not 3 or 4')
    
    x_normalized = (x - MEAN_CHANNEL_PIXELVALS[:, :n_channels, :, :])/STD_CHANNEL_PIXELVALS[:, :n_channels, :, :] # assumes the use of broadcasting to normalize all N tiles in parallel
    return x_normalized

def get_modelpath(isprocessed=False, ismultispectral=True):
    if ismultispectral: 
        n_channels = NUM_CHANNELS_MS
        if isprocessed:
            path = PROCESSED_MS_MODEL_PATH
        else: 
            path = UNPROCESSED_MS_MODEL_PATH
    else:
        n_channels = NUM_CHANNELS_RGB
        if isprocessed: 
            path = PROCESSED_RGB_MODEL_PATH
        else: 
            path = UNPROCESSED_RGB_MODEL_PATH
            
    return path, n_channels

def load_arrinet_classifier(isprocessed=False, ismultispectral=True): 
    """Classifier for tissue types of image patches 
    Dependencies: get_modelpath
    """    
    
    # Get model path and number of channels 
    modelpath, n_channels = get_modelpath(isprocessed=isprocessed, ismultispectral=ismultispectral)
    print('Loading model path: ', modelpath)

    # Initialize Arrinet
    net = densenet_av.densenet_40_12_bc(in_channels=n_channels)
    net.fc = nn.Linear(net.fc.in_features, NUM_CLASSES) # adjust last fully-connected layer dimensions
#    net = net.to(device)
    
    net.load_state_dict(torch.load(modelpath))
    net.eval()
    return net
    
def classify(x, isprocessed=False, ismultispectral=True):
    """Classifies the tissue type(s) in a stack of image patch(es) (32x32 pixel)

    Args:
        x (ndarray): multispectral image patch(es), of shape (N, C, H, W)= 
           (N, 21 or 3, 32, 32) where N is number of patches to classify in parallel
    
    Output:
        class_scores (ndarray): class scores per image patch, of shape (N, NUM_CLASSES)
        class_prob (ndarray): Prediction class probabilities, of shape (N, NUM_CLASSES) 
        pred_class_int (ndarray): Prediction class categorical index, of shape (N, 1)
        pred_class_name (list): Predication class names, of shape (N,)
        
    Dependencies: load_arrinet_classifier
    """ 
    
    #Load Arrinet
    net = load_arrinet_classifier(isprocessed=isprocessed, ismultispectral=ismultispectral)
    
    # Make input image normalized and Tensor type (if GPU desired)
    x_norm = normalize_image(x) # expects inpuut shape (N, C, H, W)
    x_norm = torch.from_numpy(x_norm) # cast numpy to torch, same scale of values because numpy array's dtype is float64, see https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ToTensor
    x_norm = x_norm.to(dtype=torch.float)  # make same dtype float as model weights
    
    with torch.no_grad(): # turn of autograd to save computational memory and speed
        # Use GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('GPU vs CPU:', device)
        net = net.to(device)
        x_norm = x_norm.to(device)
        
        # Classify - obtain class scores
#        print('Input array size:', x_norm.size(), '  tensor type:', x_norm.type())
        class_scores = net(x_norm) # torch.nn.conv2d expects input size (N, C, H, W)
        class_scores = class_scores.cpu() # copy output tensor to host memory
        
        # Obtain class probabilities
        m = nn.Softmax(dim=1)
        class_prob = m(class_scores)
        
        # Obtain highest probability class prediction
        y_pred_ind = torch.argmax(class_scores, dim=1)  # categorical class number
        pred_class_name = [TISSUE_CLASSES[i] for i in y_pred_ind] # categorical class name
        
        # Cast output variables to numpy
        class_scores =  class_scores.numpy() # cast tensor to numpy
        class_prob = class_prob.numpy()      # cast tensor to numpy
        pred_class_int = y_pred_ind.numpy()  # cast tensor to numpy
        pred_class_int = np.reshape(pred_class_int, (np.shape(pred_class_int)[0], 1)) # cast rank-1 ndarray of shape (N,) to 2D array shape (N, 1)
               
    return class_scores, class_prob, pred_class_int, pred_class_name

def main():
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #User select an image patch 
    print('Select an unprocessed, multispectral (.pkl) image patch file:')
    path_testpatch = mat.uigetfile(initialdir='C:/Users/CTLab/Documents/George/Python_data/arritissue_data/', title='Select file', filetypes=(("pickle files","*.pkl"),("all files","*.*")))
    print(path_testpatch)
    
    # Load selected image patch
    x_testpatch = pickle_loader(path_testpatch) # ndarray shape (H, W, C)
    # Make image-patch 4-D array prior to classification
    x_testpatch = np.expand_dims(x_testpatch, axis=0) # ndarray shape (N=1, H, W, C)
    x_testpatch = np.transpose(x_testpatch, axes=(0, 3, 1, 2)) # conversion of numpy array (N, H, W, C) to a shape (N, C, H, W) 
    
    # Classify image patch
    y, y_prob, y_pred_ind, y_pred_classnames = classify(x_testpatch)
    print('')
    print('Prediction class scores: ', y)
    print('Prediction class probabilities: ', y_prob)
    print('Sum of probabilities =', np.sum(y_prob, axis=1))
    print('Prediction class indices: ', y_pred_ind)
    print('Prediction class names: ', y_pred_classnames)
    
    print('Done')
    
if __name__=='__main__':
    main()