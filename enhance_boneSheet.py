# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 08:43:24 2019

@author: George

Read DICOM image. Code adapted from

Last edit: 2/26/2019 

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
import scipy
import itk
import pydicom
import skimage

def readDicom(foldername):
    print( "Reading Dicom directory: ", foldername )
    reader = sitk.ImageSeriesReader()
    
    dicom_names = reader.GetGDCMSeriesFileNames( foldername )
    reader.SetFileNames(dicom_names)
    
    image = reader.Execute()
    
    size = image.GetSize()
    print( "Image size:", size[0], size[1], size[2] )
    
    reader2 = sitk.ImageFileReader()
    first_dicom_name = os.listdir(foldername)[0]
    reader2.SetFileName(os.path.join(foldername, first_dicom_name ))
    reader2.LoadPrivateTagsOn();
    reader2.ReadImageInformation();
    for k in reader2.GetMetaDataKeys():
        v = reader2.GetMetaData(k)
        print("({0}) = = \"{1}\"".format(k,v))
    
    spacing = reader2.GetMetaData('0028|0030').split('\\') + [reader2.GetMetaData('0018|0050')]
    print('Image spacing: ', spacing)
    origin = reader2.GetMetaData('0020|0032').split('\\')
    origin = np.array(origin).astype(np.float64)
    print('Origin: ', origin)
    
    return image, size, spacing, origin

def writeDicom(image, targetfolder):    
    print( "Writing image:", targetfolder )
    
    sitk.WriteImage( image, targetfolder)
    
#def showDicom(image):
#    sitk.Show( image, "Dicom Series" , debugOn=True)
#    
    
def planar_measure(X, alpha):
    # Attempt adaptive filtering with planar measure, structure tensor T
    # Get x-gradient in "sx"
    sx = ndimage.sobel(X,axis=0,mode='constant')
    # Get y-gradient in "sy"
    sy = ndimage.sobel(X,axis=1,mode='constant')
    sz = ndimage.sobel(X, axis=2, mode='constant')
    
    # alpha  is a constant > 1. Negative certainty estimates
    # in the formula above are set to zero. A large alpha sets the
    # certainty to zero for all estimates not corresponding to the
    # plane case.
    print('Calculating Tensor matrices and planar measures....')
    gradI = np.stack([sx, sy, sz], axis=0).reshape((1,3) + X.shape)
    gradI = gradI.astype(np.float64)
    print('gradI shape: ', gradI.shape)
    T = np.multiply(np.transpose(gradI, (1,0,2,3,4)), gradI)  # Image structure tensor
    T = np.transpose(T, (2,3,4,0,1))
    lambd_unsorted = np.linalg.eigvals(T) # 4D array of shape (x.shape) + (3)
    lambd = np.sort(lambd_unsorted, axis=-1)
    print('lambd shape:', lambd.shape)

    # Calculate response function parameters
    lambd_3 = lambd[:,:,:,2] # lambd3 > labmd2 > lambd_1 >= 0
    lambd_2 = lambd[:,:,:,1]
#    lambd_1 = lambd[:,:,:,0] 
    c_plane = np.abs(lambd_3 - alpha*lambd_2)/np.abs(lambd_3) # planar measure
    c_plane[c_plane < 0] = 0
    c_plane = np.nan_to_num(c_plane)

    return c_plane
    
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
    sheet_enhancement = np.exp(-np.square(R_sheet)/(2*np.square(ALPHA)))
    blob_noise_elimination = (1 - np.exp(-np.square(R_blob)/(2*np.square(BETA))))
    noise_reduction = (1 - np.exp(-np.square(R_noise)/(2*np.square(GAMMA))))
    
    # Set behavior of undefined terms when divide by null lambd_3
    sheet_enhancement[np.isnan(sheet_enhancement)] = 1
    blob_noise_elimination[np.isnan(blob_noise_elimination)] = 0
    
    M = sheet_enhancement * blob_noise_elimination * noise_reduction
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
    
if __name__ == '__main__':
#    # ask user to select folder containing DICOM image series
#    print("Test readDicomCT.py: select DICOM folder")    
#    foldername = mat.uigetdir()
#    
#    # read image
#    image, size_image, spacing, origin = readDicom(foldername) # GetPixel indexes (x, y, z)
#    X = sitk.GetArrayFromImage(image)  # numpy array indexes (z, y, x) (height, row, column)
#    X = np.transpose(X, axes=(2,1,0)) # (z, y, x) -> (x, y, z)
#    #M_out = np.rot90(M, 1, (0,2))
#    #M_out = np.flip(M_out, axis=0)
#    print('X shape: ', X.shape)   
    
#    # Read image with pydicom
#    filepath = mat.uigetfile()
#    foldername = os.path.dirname(filepath)
#    ds = pydicom.dcmread(filepath)  # plan dataset
#    spacing2d = ds.PixelSpacing
#    spacing = [spacing2d[0], spacing2d[1], ds.SliceThickness]
#    origin = ds.ImagePositionPatient
#    size_image = [ds.Columns, ds.Rows, ds.NumberOfFrames]
#    X = ds.pixel_array
#    X = np.transpose(X, axes=(2,1,0)) # (z, y, x) -> (x, y, z)
    
    # Read NRRD image with pynrrd
    filepath = mat.uigetfile()
    foldername = os.path.dirname(filepath)
    data, header = nrrd.read(filepath)
    spacing = np.diagonal(header['space directions'])
    size_image = np.shape(data)
    origin = header['space origin']
    X = data
    
#    print('X shape: ', X.shape)  
    
    # No upsample or make isometric
    new_shape = size_image
    new_spacing = spacing
    
#    # Make image isometric and Upsample image
#    upsample_factor_iso = spacing/np.min(spacing)
#    UPSAMPLE_FACTOR = 4
#    X_old = X
#    new_shape = np.round(size_image * upsample_factor_iso * UPSAMPLE_FACTOR)
#    new_spacing = size_image/new_shape*spacing
#    X = skimage.transform.resize(X, output_shape=new_shape, order=3) # order 3 - cubic spline interpolation
    
    # Calculate sheetness measure maximum over scale sigma range
#    sigmas = np.arange(0.2,6,0.2)
    sigmas = np.arange(1,6,1)
    M = sheetness_measure(X, sigmas)    
        
    # write sheetness measure as nrrd image
    M_out = M
    filenum = 9
    filename = 'sheetness_score' + str(filenum) + '.nrrd'
    #nrrd.write(os.path.join(foldername, filename), M_out)
    nrrd.write(os.path.join(foldername, filename), M_out, detached_header=False, header={'sizes': new_shape, 'spacings': new_spacing})
    #nrrd.write(os.path.join(foldername, filename), M_out, detached_header=False, header={'sizes': size_image, 'spacings': spacing, 'space origin':nrrd.format_optional_vector(origin)})
        
    # write sheetness-boosted CT nrrd image
    BOOST=20000
    X_boosted = X + BOOST*M
    filename2 = 'zbone_Descoteaux_boost' + '_' + str(BOOST) + '.nrrd'
    #nrrd.write(os.path.join(foldername, filename), M_out)
    nrrd.write(os.path.join(foldername, filename2), X_boosted, header={'sizes': new_shape, 'spacings': new_spacing})
    
    # Calculate planar measure of image
    ALPHA = 2
    BETA = 420
    c_plane = planar_measure(X, ALPHA)
    nrrd.write(os.path.join(foldername, filename), c_plane, detached_header=False, header={'sizes': new_shape, 'spacings': new_spacing})

    
    # write planar-boosted CT nrrd image
    X_planar_boosted = X + BETA*c_plane
    filename3 = 'zbone_Westin_boost' + str(ALPHA) + '_' + str(BETA) + '.nrrd'
    nrrd.write(os.path.join(foldername, filename3), X_planar_boosted, header={'sizes': new_shape, 'spacings': new_spacing})

    # write sheetness- and planar-boosted CT nrrd image
    X_both_boosted = X + BETA*c_plane + BOOST*M
    filename4 = 'zbone_Descoteaux_Westin_boost' + str(ALPHA) + '_' + str(BETA) + '_' + str(BOOST) + '.nrrd'
    nrrd.write(os.path.join(foldername, filename4), X_both_boosted, header={'sizes': new_shape, 'spacings': new_spacing})

    
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
