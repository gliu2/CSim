# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 12:07:55 2019

Data loader for ARRIScope multispectral tile in pickle files

@author: CTLab
"""
import pickle
import os
import torch
import torchvision

def pickle_loader(input_file):
    item = pickle.load(open(input_file, 'rb'))
    return item.values

#test_data= torchvision.datasets.DatasetFolder(root='.', loader=pickle_loader, extensions='.pkl', transform=transform)
data_dir = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/patches3d'
test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False)

#%% Read pickle

out_path = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/output'
#fname = 'lossacc_nclass12_Truepretrained_batchsize64_epochs25lr0.001dec7_0.1_val.pkl'
fname = 'lossacc_nclass12.pkl'
[cache_loss, cache_acc] = pickle.load(open(os.path.join(out_path, fname), "rb"))
