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
    
    # Attempt adaptive filtering with planar measure, structure tensor T
    # Get x-gradient in "sx"
    sx = ndimage.sobel(X,axis=0,mode='constant')
    # Get y-gradient in "sy"
    sy = ndimage.sobel(X,axis=1,mode='constant')
    sz = ndimage.sobel(X, axis=2, mode='constant')
    
#    IxIx = np.multiply(sx,sx)
#    IxIy = np.multiply(sx, sy)
#    IxIz = np.multiply(sx,sz)
#    IyIy = np.multiply(sy,sy)
#    IyIz = np.multiply(sy,sz)
#    IzIz = np.multiply(sz,sz)
    c_plane = np.zeros(np.shape(X)) # planar measure
    # alpha  is a constant > 1. Negative certainty estimates
    # in the formula above are set to zero. A large alpha sets the
    # certainty to zero for all estimates not corresponding to the
    # plane case.
    alpha = 3     # weight factor of planar measure
    beta = 420 # planar measure threshold effect constant
    print('Calculating Tensor matrices and planar measures....')
    for i in np.arange(X.shape[0]):
        for j in np.arange(X.shape[1]):
            for k in np.arange(X.shape[2]):
                T = np.zeros((3,3))
                gradI = np.zeros((1,3))
                gradI[0,0] = sx[i,j,k]
                gradI[0,1] = sy[i,j,k]
                gradI[0,2] = sz[i,j,k]
                T = np.multiply(gradI.T, gradI)
#                T[0,0] = IxIx[i][j][k]
#                T[1,0] = IxIy[i][j][k]
#                T[0,1] = T[1,0]
#                T[2,0] = IxIz[i][j][k]
#                T[0,2] = T[2,0]
#                T[1,1] = IyIy[i][j][k]
#                T[1,2] = IyIz[i][j][k]
#                T[2,1] = T[1,2]
#                T[2,2] = IzIz[i][j][k]
                lambd, eigenvectors = np.linalg.eig(T)
#                lambd_asc = lambd.sort() # eigenvalues in ascending order
#                print('lambd_asc: ', lamb)
                ord_ind = np.argsort(lambd)
                lambd_3 = lambd[ord_ind[2]]
                lambd_2 = lambd[ord_ind[1]]
                lambd_1 = np.min(lambd) # lambd3 > labmd2 > lambd_1 >= 0
                c_plane[i,j,k] = (lambd_3 - alpha*lambd_2)/lambd_3
                if ((lambd_3 - alpha*lambd_2)/lambd_3) < 0:
                    c_plane[i,j,k] = 0
                    
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
    bone_Westin_boosted = np.rot90(bone_Westin_boosted, 1, (0,2))
    bone_Westin_boosted = np.flip(bone_Westin_boosted, axis=0)
    filename3 = 'bone_Westin_boost' + str(alpha) + '_' + str(beta) + '.nrrd'
    nrrd.write(os.path.join(foldername, filename3), bone_Westin_boosted)
    
    # show image
#    showDicom(image)
##    plt.imshow(X, cmap='gray')