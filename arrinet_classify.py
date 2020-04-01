# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:18:04 2020

Classify multispectral 21-channel image patches (32x32)


@author: George Liu

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

NUM_CHANNELS = 21 # number of multispectral image channels
NUM_CLASSES = 11 # cartilage and perichondrium are joined into 1 class

def pickle_loader(input_file):
    item = pickle.load(open(input_file, 'rb'))
    return item

def normalize_image(x):
    n_channels = np.shape(x)[-1]
    return (x - MEAN_CHANNEL_PIXELVALS[:n_channels])/STD_CHANNEL_PIXELVALS[:n_channels]

def get_modelpath(isprocessed=False, ismultispectral=True):
    if isprocessed:
        if ismultispectral: return PROCESSED_MS_MODEL_PATH
        else: return PROCESSED_RGB_MODEL_PATH
    else:
        if ismultispectral: return UNPROCESSED_MS_MODEL_PATH
        else: return UNPROCESSED_RGB_MODEL_PATH

def load_arrinet_classifier(isprocessed=False, ismultispectral=True): 
    # Classifier for tissue types of image patches 
    # Dependencies: get_modelpath
        
    # Initialize Arrinet
    net = densenet_av.densenet_40_12_bc(in_channels=NUM_CHANNELS)
    net.fc = nn.Linear(net.fc.in_features, NUM_CLASSES) # adjust last fully-connected layer dimensions
#    net = net.to(device)
    
    modelpath = get_modelpath(isprocessed=isprocessed, ismultispectral=ismultispectral)
    print(modelpath)
    net.load_state_dict(torch.load(modelpath))
    net.eval()
    return net
    
def classify(x, isprocessed=False, ismultispectral=True):
    # Classify tissue type in one image patch (32x32 pixel)
    # x - multispectral image patch, matrix of shape (h, w, c)= (32, 32, 21)
    #
    ## Dependencies: load_arrinet_classifier
    
    #Load Arrinet
    net = load_arrinet_classifier(isprocessed=False, ismultispectral=True)
    
    # Make input image normalized and Tensor type (if GPU desired)
    x_norm = normalize_image(x)
    x_norm = torch.from_numpy(np.transpose(x_norm, axes=(2,0,1))) # conversion of numpy array (H x W x C) to a torch.FloatTensor of shape (C x H x W) with same scale because numpy array's dtype is float64, see https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.ToTensor
    x_norm = torch.unsqueeze(x_norm, dim=0) # change shape to (1, C, H, W)
#    x_norm = x_norm.repeat(3,1,1,1) # test that multiple images can be classified in parallel ie N>1
    x_norm = x_norm.to(dtype=torch.float)  # make same dtype float as model weights
    
    # Classify
    print('Input array size:', x_norm.size(), '  tensor type:', x_norm.type())
    output = net(x_norm) # torch.nn.conv2d expects input size (N, C, H, W)
    return output

def main():
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #User select an image patch to load
    print('Select an unprocessed, multispectral (.pkl) image patch file:')
    path_testpatch = mat.uigetfile(initialdir='C:/Users/CTLab/Documents/George/Python_data/arritissue_data/', title='Select file', filetypes=(("pickle files","*.pkl"),("all files","*.*")))
    print(path_testpatch)
    
    x_testpatch = pickle_loader(path_testpatch)
    y = classify(x_testpatch)
    print('')
    print('Prediction class scores: ', y.tolist())
    
    # Obtain class probabilities
    m = nn.Softmax(dim=1)
    y_prob = m(y)
    print('Prediction class probabilities: ', y_prob.tolist())
    print(' Sum of probabilities =', torch.sum(y_prob, dim=1).tolist())
    
    y_pred_ind = torch.argmax(y, dim=1)
    print('Prediction class indices: ', y_pred_ind.tolist())
    y_pred_classnames = [TISSUE_CLASSES[i] for i in y_pred_ind]
    print('Prediction class names: ', y_pred_classnames)
    print('Done')
    
if __name__=='__main__':
    main()