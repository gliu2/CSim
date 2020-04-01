# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:18:04 2020

Classify multispectral 21-channel image patches (32x32)


@author: George Liu

Dependencies: mat.py, densenet_av.py, plot_confusion_matrix.py
"""

import numpy as np
import mat

NUM_CHANNELS = 21 # number of multispectral image channels

def 

def classify(x): 
    # Use 
    # x - multispectral image patch, matrix of shape (h, w, c)= (32, 32, 21)
    
    # Initialize Arrinet
    net = densenet_av.densenet_40_12_bc(in_channels=NUM_CHANNELS)
    net.fc = nn.Linear(net.fc.in_features, NUM_CHANNELS) # adjust last fully-connected layer dimensions
    net = net.to(device)
    
    net.load_state_dict(torch.load(modelpath))
    net.eval()
    
    
def main():
    
    
if __name__=='__main__':
    main()