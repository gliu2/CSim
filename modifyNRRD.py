# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:53:38 2019

@author: George

Modify NRRD image. Code adapted from

Last edit: 2/10/2019 

Dependencies: mat.py
"""

#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import SimpleITK as sitk
import sys, os
import mat
import matplotlib.pyplot as plt
import nrrd
from scipy import ndimage

def writeDicom(image, targetfolder):    
    print( "Writing image:", targetfolder )
    
    sitk.WriteImage( image, targetfolder)
    
#def showDicom(image):
#    sitk.Show( image, "Dicom Series" , debugOn=True)
#    

if __name__ == '__main__':
    
    # ask user to select folder containing DICOM image series
    print("Test modifyNRRD.py: select NRRD file")    
    file_path = mat.uigetfile()
    
    # read image
    X, header = nrrd.read(file_path) # data, assume shape is (H,W,D) <--(X,Y,Z) with horizontal flip
    print(X.shape)    # numpy array indexes (z, y, x) (height, row, column)
    
    # flip image horizontally (R-L), trial 1
    print("Flip images....")
#    X_flip1 = np.fliplr(X) # flips axies 1
#    X_flip2 = np.flip(X, axis=2)      
    X_flip0 = np.flip(X, axis=0)
    
    # flip image horizontally, trial 2
    
    # get directory name
    folder_path = os.path.dirname(os.path.abspath(file_path))
    
    # write modified image as NRRD and DICOM
    filename1 = 'ethmoydbone_Westin_boost2_420_flip0.nrrd'
#    filename2 = 'ethmoydbone_Westin_boost2_420_flip2.nrrd'
#    filename3 = 'ethmoydbone_Westin_boost2_420_flip1.nrrd'
#    filename4 = 'ethmoydbone_Westin_boost2_420_flip1.nrrd'    
    
    print("Writing first NRRD file....")
    nrrd.write(os.path.join(folder_path, filename1), X_flip0)
#    print("Writing second NRRD file....")
#    nrrd.write(os.path.join(folder_path, filename2), X_flip2)
#    
#    dicom_image = sitk.GetImageFromArray(X_flip1)
#    
#    sitk.WriteImage(X_flip1, os.path.join(folder_path, filename3))
#    sitk.WriteImage(X_flip2, os.path.join(folder_path, filename4))