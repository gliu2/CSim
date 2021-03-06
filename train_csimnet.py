# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 11:04:55 2018

@author: George Liu

Train fully convolutional neural network (CSimNet) to automatically segment temporal bone CT images.

Input: paths to directories containing CT image files of:
        - Training images (original CT scans)
        - Training labels (manually-segmented masks of desired structures)
        - more tbd...
Output: Saves following variables in pickle file 'GL_3dunet_train_loss.pkl':
            cache_loss - list of tuples (t, loss), where 't' is mini-batch # and 'loss' is training cross-entropy loss
    
Dependencies: Unet3d_pytorchGL.py
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
import operator
import pickle

# Training settings
BATCH_SIZE = 4

parser = argparse.ArgumentParser(description="PyTorch CSim_segmentation")
parser.add_argument("--how2normalize", type=int, default=0, help="how to normalize the data")
#parser.add_argument("--batchSize", type=int, default=10, help="training batch size")
parser.add_argument("--batchSize", type=int, default=BATCH_SIZE, help="training batch size") # reduce batch size for testing
parser.add_argument("--numofIters", type=int, default=200000, help="number of iterations to train for")
parser.add_argument("--showTrainLossEvery", type=int, default=1, help="number of iterations to show train loss")
parser.add_argument("--saveModelEvery", type=int, default=4000, help="number of iterations to save the model")
parser.add_argument("--showTestPerformanceEvery", type=int, default=1, help="number of iterations to show test performance")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--decLREvery", type=int, default=100000, help="Sets the learning rate to the initial LR decayed by momentum every n iterations, Default: n=40000")
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
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
# Note: 3D unet input 132 x 132 x 116 voxels; Vnet input 128 x 128 x 64 voxels
#tile_H = 132
#tile_W = 132
#tile_D = 116

#tile_H = 128
#tile_W = 128
#tile_D = 112

tile_H = 64
tile_W = 64
tile_D = 56

#tile_H = 3
#tile_W = 3
#tile_D = 3

def Generator_3D_patchesGL(path_folder, batchsize):
    # Generator for batch of patches
    # Dependencies: Generator_3D_patch
    patch_generator = Generator_3D_patch(path_folder)
    next_batch = np.zeros((batchsize,tile_H,tile_W,tile_D))
    while True:
        for n in range(0,batchsize):
            next_batch[n] = next(patch_generator)
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
            print('   Working on:', namepatient)
            filename = os.path.join(path_folder, namepatient)
            data, header = nrrd.read(filename) # training data, assume shape is (H,W,D) <--(X,Y,Z)
#            data = np.reshape(np.arange(0,1000), (10,10,10)) # pretend data, use tile_H = 3, tile_W = 3, tile_Z = 3
#            print('Original CT shape: ', data.shape)
            
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
            
def debug_memory():
    import collections, gc, torch
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    for line in sorted(tensors.items(), key=operator.itemgetter(1)):
        print('{}\t{}'.format(*line))
            

def main():    
    
    global opt, model 
    opt = parser.parse_args()

    # Check if CUDA is available
    opt.device = None
    if not opt.disable_cuda and torch.cuda.is_available():
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')
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

#    path_patients_X = '/Users/George/Documents/Blevins/CT_Segmentations/original_CT/'  # for GL windows machine
#    path_patients_y = '/Users/George/Documents/Blevins/CT_Segmentations/CarotidArtery/'  # for GL windows machine
    path_patients_X = 'C:/Users/CTLab/Documents/George/CT_segmentations/original_CT'  # for CSim lab windows machine
    path_patients_y = '/Users/CTLab/Documents/George/CT_Segmentations/CarotidArtery/'   # for CSim lab windows machine

#    batch_size=10
#    data_generator = Generator_3D_patches(path_patients_h5,opt.batchSize,inputKey1='dataMR',outputKey='dataSeg')
#    data_generator = Generator_3D_patches(path_patients_X, path_patients_y, opt.batchSize) # utils.py
    print('Batch size:', opt.batchSize) 
    
    print('Instantiate generators....')
    batch_generatorGL = Generator_3D_patchesGL(path_patients_X, opt.batchSize)
    label_generatorGL = Generator_3D_patchesGL(path_patients_y, opt.batchSize) # local 
    print('Finished instantiating generators.')

    # Construct our model by instantiating the class defined in Unet3d_pytorch.py
    net = UNet3D_GL(in_channel=1, n_classes=2) # kernel_size=3, stride=1, padding=1, bias=False, batchnorm=False      
    net = net.to(opt.device)
        
    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr=opt.lr)
    print('Start training model...')
    cache_loss = []
    for t in range(1000):
       
        # empty CUDA cache every few minibatches
        if opt.device == torch.device('cuda') and np.mod(t,2)==0:
            torch.cuda.empty_cache()
            
        # Generate next mini-batch
#        print('Obtain next training batch for iteration: ', t)
        next_batch = next(batch_generatorGL)
#        print('Obtain next label batch for iteration: ', t)
        next_batch_y = next(label_generatorGL)
        y = torch.from_numpy(next_batch_y)
        x_expanded = np.expand_dims(next_batch, axis=1)
#        print('Shape first batch:', next_batch.shape)
#        print('Shape first batch expanded:', x_expanded.shape)
    #    x_expanded = x_expanded.float()
        x = torch.from_numpy(x_expanded).float() # match float default type for weights and biases 
        
        # Convert target labels to type long for computing loss function
        y = y.type(dtype=torch.long)
        
        #  pushing tensors to CUDA device if available (you have to reassign them)
        x = x.to(opt.device)
        y = y.to(opt.device)
        
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = net(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        print(t, loss.item())
        
        # Save losses to plot learning curve
        cache_loss.append((t, loss.item()))

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
    
        # Analyze memory footprint to debug CPU/CUDA memory leaks
#        debug_memory()
        
    # Save cached variables to analyze training / learning curve
    with open('GL_3dunet_train_loss.pkl', 'wb') as ff:
        pickle.dump(cache_loss, ff)
    
    print('Done')
    
if __name__ == '__main__':     
    main()
    
    
## Getting back the objects:
#with open('GL_3dunet_train_loss.pkl', 'rb') as ff:  
#    cache_loss = pickle.load(ff)