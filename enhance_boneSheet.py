# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 08:43:24 2019

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
import itk


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
    
def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray (y,x)
    Returns:
       an array of shape (x.ndim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x) # list of length x.ndim containing arrays of x.shape
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype) 
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k) 
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian
    
def hessian_scale2d(x, sigma):
    """
    Calculate the hessian matrix with derivative of Gaussian kernel convolutions
    Parameters:
       - x : ndarray (z,y,x)
       - sigma : scale, standard deviation of Gaussian kernel
    Returns:
       an array of shape (x.ndim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
#    Ixx = ndimage.gaussian_filter(x, sigma, order=(0,2)) 
#    Iyy = ndimage.gaussian_filter(x, sigma, order=(2,0)) 
#    Ixy = ndimage.gaussian_filter(x, sigma, order=(1,1)) # 1st derivative of gaussian in x-direction
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    # upper triangular matrix of unique hessian values
    for i in np.arange(x.ndim):
        for j in np.arange(x.ndim-1, i-1, -1):
            hessian[i,j,:,:] = ndimage.gaussian_filter(x, sigma, order=(2-i-j,i+j))
            hessian[j,i,:,:] = hessian[i,j,:,:] # make hessian matrix symmetrical
    return hessian
    
def hessian_scale3d(x, sigma):
    """
    Calculate the hessian matrix with derivative of Gaussian kernel convolutions
    Parameters:
       - x : ndarray
       - sigma : scale, standard deviation of Gaussian kernel
    Returns:
       an array of shape (x.ndim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    # upper triangular matrix of unique hessian values
    for i in np.arange(x.ndim):
        for j in np.arange(x.ndim-1, i-1, -1):
            z_order = int(i==0) + int(j==0)
            y_order = int(i==1) + int(j==1)
            x_order = int(i==2) + int(j==2)
            hessian[i,j,:,:] = ndimage.gaussian_filter(x, sigma, order=(z_order,y_order, x_order))
            if i==j:
                hessian[j,i,:,:] = hessian[i,j,:,:] # make hessian matrix symmetrical
    return hessian
    
def sheetness_response(x, sigma):
    """
    Calculate the sheetness response of 3d image over input scale, as defined
    by Descoteau et al. 2006
    Parameters:
       - x : ndarray (z,y,x)
       - sigma : scale, standard deviation of Gaussian kernel
    Returns:
       an array of shape x.shape
       where the array[i, j, k] corresponds to the sheetness measure 
    """
    # Calculate hessian matrix over input scale
    hessian = hessian_scale3d(x, sigma)
    
    # Transpose hessian matrix values to last two dimensions of array
    hessian_transpose = np.transpose(hessian, axes=(2,3,4,0,1))
    
    # Vectorized eigenvalue decomposition of Hessian matrices
    lambd_unsorted = np.linalg.eigvals(hessian_transpose) # 4D array of shape (x.shape) + (3)
    lambd = np.sort(lambd_unsorted, axis=-1)

    # Calculate response function parameters
    lambd_3 = lambd[:,:,:,2] # lambd3 > labmd2 > lambd_1 >= 0
    lambd_2 = lambd[:,:,:,1]
    lambd_1 = lambd[:,:,:,0] 
    
    R_sheet = np.abs(lambd_2)/np.abs(lambd_3)
    R_blob = np.abs(2*np.abs(lambd_3) - np.abs(lambd_2)-np.abs(lambd_1))/np.abs(lambd_3)
    R_noise = np.sqrt(np.square(lambd_1) + np.square(lambd_2) + np.square(lambd_3))
    
    # Calculate sheetness measure for scale sigma: M = {0 if lambd_3>0, [...] otherwise}
    # Use parameters from Descoteaux et al. 2006 paper
    ALPHA = 0.5
    BETA = 0.5
    GAMMA = 0.5*np.amax(R_noise)
    
    M = np.empty(x.shape, dtype=x.dtype)
    M = np.exp(-np.square(R_sheet)/(2*np.square(ALPHA))) * \
        (1 - np.exp(-np.square(R_blob)/(2*np.square(BETA)))) * \
        (1 - np.exp(-np.square(R_noise)/(2*np.square(GAMMA))))
    M[lambd_3>0] = 0
    M = np.nan_to_num(M)    # replace nan values with zero
    return M
    
def sheetness_measure(x, sigmas):
    """
    Calculate the sheetness measure of 3d image, maximum sheetness response 
    over all input scales as defined by Descoteau et al. 2006
    Parameters:
       - x : ndarray (z,y,x)
       - sigmas : vector set of scales
    Returns:
       an array of shape x.shape
       where the array[i, j, k] corresponds to the sheetness measure M at (i,j,k)
    Dependencies: sheetness_response(x, sigma)
    """
    M = np.zeros(x.shape, dtype=x.dtype)
    for i, sigma in enumerate(sigmas):
        M_current = sheetness_response(x, sigma)
        M = np.maximum(M, M_current)
        print('Sigma: ', sigma)
        print(M[:3,:3,1])
    return M
    
#if __name__ == '__main__':
# ask user to select folder containing DICOM image series
print("Test readDicomCT.py: select DICOM folder")    
foldername = mat.uigetdir()

# read image
image = readDicom(foldername) # GetPixel indexes (x, y, z)
X = sitk.GetArrayFromImage(image) 
print(X.shape)    # numpy array indexes (z, y, x) (height, row, column)

# Calculate sheetness measure maximum over scale sigma range
sigmas = np.arange(1,6,0.5)
M = sheetness_measure(X, sigmas)    
    
# write sheetness measure as nrrd image
M_out = np.rot90(M, 1, (0,2))
M_out = np.flip(M_out, axis=0)
filename = 'sheetness_score3' + '.nrrd'
nrrd.write(os.path.join(foldername, filename), M_out)
    


#### Test sheetness_measure
## Calculate sheetness measure maximum over scale sigma range
#X = np.random.randn(10,10,10)
#sigmas = [1, 1.5, 2]
#M = sheetness_measure(X, sigmas)

        
#### Test sheetness_response
#X = np.random.randn(100,100,100)
#
## Calculate sheetness measure over scale sigma
#sigma = 1
#sheetness = sheetness_response(X, sigma)        
        
        
#    # Broken ITK sheetness filter
#    sheetness = itk.DescoteauxSheetnessImageFilter.New(X)
#    M = sitk.DescoteauxSheetnessImageFilter(X) # https://itk.org/Doxygen/html/classitk_1_1DescoteauxSheetnessImageFilter.html
        
    
##### Test hessian_scale2d
##x = np.random.randn(100, 100, 100)
##x = np.array([[1, 2, 6], [3, 4, 5]])
#x = np.zeros((10,10))
#x[:,6:] = 9
#x[:, 4] = 1
#x[:, 5] = 4
#y = hessian(x)        
#sigma = 1
#
#h2 = hessian_scale2d(x, sigma)
#Ixx = ndimage.gaussian_filter(x, sigma, order=(0,2)) 
#Iyy = ndimage.gaussian_filter(x, sigma, order=(2,0)) 
#Ixy = ndimage.gaussian_filter(x, sigma, order=(1,1)) # 1st derivative of gaussian in x-direction
#
#
    
##### Test hessian_scale3d
#x2 = np.zeros((10,10,10))
#x2[:,:,6:] = 9
#x2[:,:, 4] = 1
#x2[:,:, 5] = 4

#h3 =hessian_scale3d(x2,sigma)
