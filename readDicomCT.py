# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:53:38 2019

@author: George

Read DICOM image. Code adapted from

Last edit: 2/3/2019 

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
import enhance_boneSheet


def readDicom(foldername):
    print( "Reading Dicom directory: ", foldername )
    reader = sitk.ImageSeriesReader()
    
    dicom_names = reader.GetGDCMSeriesFileNames( foldername )
    reader.SetFileNames(dicom_names)
    
    image = reader.Execute()
    
    size = image.GetSize()
    print( "Image size:", size[0], size[1], size[2] )
    
    return image

def writeDicom(image, targetfolder):    
    print( "Writing image:", targetfolder )
    
    sitk.WriteImage( image, targetfolder)
    
#def showDicom(image):
#    sitk.Show( image, "Dicom Series" , debugOn=True)
#    

if __name__ == '__main__':
    # ask user to select folder containing DICOM image series
    print("Test readDicomCT.py: select DICOM folder")    
    foldername = mat.uigetdir()
    
    # read image
    image = readDicom(foldername) # GetPixel indexes (x, y, z)
    X = sitk.GetArrayFromImage(image) 
    X = np.transpose(X, axes=(2,1,0)) # (z, y, x) -> (x, y, z)
    print(X.shape)    # numpy array indexes (z, y, x) (height, row, column)
    
    # thresshold soft tissue
    t_soft = -150 # soft tissue threshold (HU)
    t_bone = 1000 # bone threshold (HU)
    
#    air = X < t_soft
#    bone = X > t_bone
#    soft = ~np.logical_or(air, bone)
#    
#    # write image of bone
##    image_bone = sitk.GetImageFromArray(bone.astype(int))
#    bone = bone.astype(int)
#    bone_image = np.rot90(bone, 1, (0,2))
#    filename = 'bone_seg8.nrrd'
#    nrrd.write(foldername + filename, bone_image)
    
    # alpha  is a constant > 1. Negative certainty estimates
    # in the formula above are set to zero. A large alpha sets the
    # certainty to zero for all estimates not corresponding to the
    # plane case.
    alpha = 2     # weight factor of planar measure
    beta = 420 # planar measure threshold effect constant
    c_plane = enhance_boneSheet.planar_measure(X, alpha)
                    
#    # Adaptively threshold with planar measure
#    t_0 = t_bone # global threshold
#    t = t_0 - beta*c_plane # adaptive reshold based on Westin et al.
#    bone_Westin = np.greater(X, t)
#    # write image of bone
#    bone_Westin = bone_Westin.astype(int)
#    bone_image_Westin = np.rot90(bone_Westin, 1, (0,2))
#    filename2 = 'bone_Westin_seg' + str(beta) + '.nrrd'
#    nrrd.write(os.path.join(foldername, filename2), bone_image_Westin)
    
    # boost nrrd image
    bone_Westin_boosted = X + beta*c_plane
#    bone_Westin_boosted = np.rot90(bone_Westin_boosted, 1, (0,2))
#    bone_Westin_boosted = np.flip(bone_Westin_boosted, axis=0)
    filename3 = 'bone_Westin_boost' + str(alpha) + '_' + str(beta) + '.nrrd'
    nrrd.write(os.path.join(foldername, filename3), bone_Westin_boosted)
    
    # show image
#    showDicom(image)
##    plt.imshow(X, cmap='gray')