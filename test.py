# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 11:04:55 2018

@author: George

Test how utils.py Generator_3D_patches
"""

# from __future__ import print_function
import argparse, os
import torch.nn as nn   # install at https://pytorch.org/
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import torch
import torch.utils.data as data_utils
#from utils import *
#from Unet2d_pytorch import UNet
#from Unet3d_pytorch import UNet3D
#from nnBuildUnits import CrossEntropy3d
#import time
import nrrd   # install at https://pypi.org/project/pynrrd/
from Unet3d_pytorchGL import UNet3D_GL

# Training settings
parser = argparse.ArgumentParser(description="PyTorch InfantSeg")
parser.add_argument("--how2normalize", type=int, default=0, help="how to normalize the data")
#parser.add_argument("--batchSize", type=int, default=10, help="training batch size")
parser.add_argument("--batchSize", type=int, default=1, help="training batch size") # reduce batch size for testing
parser.add_argument("--numofIters", type=int, default=200000, help="number of iterations to train for")
parser.add_argument("--showTrainLossEvery", type=int, default=1, help="number of iterations to show train loss")
parser.add_argument("--saveModelEvery", type=int, default=4000, help="number of iterations to save the model")
parser.add_argument("--showTestPerformanceEvery", type=int, default=1, help="number of iterations to show test performance")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--decLREvery", type=int, default=100000, help="Sets the learning rate to the initial LR decayed by momentum every n iterations, Default: n=40000")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--prefixModelName", default="Segmentor_wce_lrdcr_0118_", type=str, help="prefix of the to-be-saved model name")
parser.add_argument("--prefixPredictedFN", default="preSub_wce_lrdcr_0118_", type=str, help="prefix of the to-be-saved predicted filename")

# Create samples that is multiple of batchsize
# Cicek et al. 2016 used input voxel tile shape of (132,132,116)
tile_H = 132
tile_W = 132
tile_D = 116

#tile_H = 3
#tile_W = 3
#tile_D = 3

def Generator_3D_patchesGL(path_folder, batchsize):
    # Generator for batch of patches
    # Dependencies: Generator_3D_patch
    patch_generator = Generator_3D_patch(path_folder)
    next_batch = np.zeros((batchsize,tile_H,tile_W,tile_D))
    for n in range(0,batchsize):
        next_batch[n] = patch_generator.__next__()
    yield next_batch    
    
    
def Generator_3D_patch(path_folder): 
    # Generator for iterable that yields patches of (X[i], y[i]), where X is 
    # original image (training data), y segmented image (segmented image), and
    # patch i is obtained by 
    # Input: path to folder with CT nrrd files of shape: (H,W,D), 1 channel
    # Output: generator for a batch of N patches at a time, in output matrix: (N,H,W,D).
    # The output matrix will eventually need to be reshaped to (N,C=1,D,H,W) for input to Unet3D for 
    #   conv3D and ConvTranspose3d input shape
    patients = os.listdir(path_folder) # directory of CT nrrd files
    while True:
        for idx,namepatient in enumerate(patients):
            # Read CT image for patient idx
            print('Working on:', namepatient)
            filename = os.path.join(path_folder, namepatient)
            data, header = nrrd.read(filename) # training data, assume shape is (H,W,D) <--(X,Y,Z)
#            data = np.reshape(np.arange(0,1000), (10,10,10)) # pretend data, use tile_H = 3, tile_W = 3, tile_Z = 3
            print('Original CT shape: ', data.shape)
            
            # Extrapolate input data at edges of CT to accomodate tiling
            toadd_x = tile_H - np.mod(data.shape[0], tile_H)
            toadd_y = tile_W - np.mod(data.shape[1], tile_W)
            toadd_z = tile_D - np.mod(data.shape[2], tile_D)
            data_padded = np.pad(data, ((0,toadd_x),(0,toadd_y),(0,toadd_z)), 'symmetric')
            
            # Calculate number of sample tiles in CT
            N_x = data_padded.shape[0]/tile_H
            N_y = data_padded.shape[1]/tile_W
            N_z = data_padded.shape[2]/tile_D
            N = N_x * N_y * N_z
            if np.mod(N,1)!=0:
                print('Warning: Number of tiles', N, 'is not integer')
            
            # Generate next sample tile
#            batch_tiles = np.zeros(batchsize, tile_H, tile_W, tile_D)
            for i in range(0, int(N_x)):
                for j in range(0, int(N_y)):
                    for k in range(0, int(N_z)):
#                        print('(i,j,k):', i, j, k)
                        next_tile = data_padded[i*tile_H:(i+1)*tile_H, j*tile_W:(j+1)*tile_W, k*tile_D:(k+1)*tile_D]
                        yield next_tile
            # For debugging, yield just first tile from each CT image
#            for j in range(0,batchsize)
#                sample = data[:tile_H,:tile_W,:tile_D]
#            yield tile1
            
            

def main():    
    
    global opt, model 
    opt = parser.parse_args()
    print(opt)

#    optimizer = optim.SGD(net.parameters(),lr=opt.lr)
#    criterion = nn.CrossEntropyLoss()
#    net = UNet3D(in_channel=1, n_classes=2) # kernel_size=3, stride=1, padding=1, bias=False, batchnorm=False
#    #net.cuda()
#    params = list(net.parameters())
#    print('len of params is ')
#    print(len(params))
#    print('size of params is ')
#    print(params[0].size())

    path_patients_X = '/Users/George/Documents/Blevins/CT_Segmentations/original_CT/'
    path_patients_y = '/Users/George/Documents/Blevins/CT_Segmentations/CarotidArtery/'

#    batch_size=10
#    data_generator = Generator_3D_patches(path_patients_h5,opt.batchSize,inputKey1='dataMR',outputKey='dataSeg')
#    data_generator = Generator_3D_patches(path_patients_X, path_patients_y, opt.batchSize) # utils.py
    print('Batch size:', opt.batchSize) 
    
#    data_generatorGL = Generator_3D_patchesGL(path_patients_X, opt.batchSize) # local 
#    data_generatorGL_test = Generator_3D_patchesGL(path_patients_y, opt.batchSize)
   
#    # Test Generator_3D_patch
#    print('Test Generator_3D_patch')
#    data_generatorGL_patch = Generator_3D_patch(path_patients_X)
#    N = 5
#    X = np.zeros((N,tile_H, tile_W, tile_D))
#    for i in range(0,N):
#        print('Working on tile:', i+1, 'of', N)
#        data_batch = data_generatorGL_patch.__next__() # Returns 1 data tile
#        X[i,:,:,:] = data_batch
#        print(data_batch)
#        
#    # Test Generator_3D_patchesGL
#    print('Test Generator_3D_patchesGL')
#    data_generatorGL = Generator_3D_patchesGL(path_patients_X, opt.batchSize) # local 
#    N2 = 1 # number of batches to generate
#    X2 = np.zeros((N2*opt.batchSize,tile_H, tile_W, tile_D))
##    y = np.zeros((N,tile_H, tile_W, tile_D))
#    for i in range(0,N2):
#        print('Working on batch:', i+1, 'of', N2, 'of size', opt.batchSize)
#        data_batch = data_generatorGL.__next__() # Returns 1 data tile
#        X2[i*opt.batchSize:(i+1)*opt.batchSize,:,:,:] = data_batch
#        print('X2 shape:', X2.shape)
#        print(X2)   
        
    # Train unit with toy data
    # Note: 3D unet input 132 x 132 x 116 voxels; Vnet input 128 x 128 x 64 voxels
    # Create random Tensors to hold inputs and outputs
    batch_generatorGL = Generator_3D_patchesGL(path_patients_X, opt.batchSize)
    first_batch = batch_generatorGL.__next__()
    x_expanded = np.expand_dims(first_batch, axis=1)
    print('Shape first batch:', first_batch.shape)
    print('Shape first batch expanded:', x_expanded.shape)
#    x_expanded = x_expanded.float()
    x = torch.from_numpy(x_expanded).float() # match float default type for weights and biases 
    label_generatorGL = Generator_3D_patchesGL(path_patients_y, opt.batchSize) # local 
    y = torch.from_numpy(label_generatorGL.__next__())

    # Construct our model by instantiating the class defined in Unet3d_pytorch.py
    net = UNet3D_GL(in_channel=1, n_classes=2) # kernel_size=3, stride=1, padding=1, bias=False, batchnorm=False        
        
    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr=opt.lr)
    print('Start training model...')
    for t in range(500):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = net(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    
    #    inputs,labels = data_generator.__next__() # 132 x 132 x 116 voxel tile of the image    
#        print('shape is ....',inputs.shape)
#        labels = np.squeeze(labels)
#        labels = labels.astype(int)
#        print(labels.shape)
#        inputs = np.transpose(inputs,(0,4,2,3,1))
#        labels = np.transpose(labels,(0,3,2,1))
#        inputs = torch.from_numpy(inputs)
#        labels = torch.from_numpy(labels)
#    print('X:',X)
#    print('y:',y)
    print('Done')
    
if __name__ == '__main__':     
    main()
    