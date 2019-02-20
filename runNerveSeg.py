# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 11:05:44 2018

@author: George S. Liu

Train a 3-D U-Net on labeled CT scans of the temporal bone. Goal is semantic 
segmentation of carotid artery, facial nerve, inner ear, ossicles, sigmoid sinus.

Cardinal Sim project with Dr. Blevins

Last edit: 10/13/2018

Dependencies: utils.py, Unet2d_pytorch.py, Unet3d_pytorch.py, nnBuildUnits.py (by Dong Nie)
Modules: PyTorch (torch), nrrd
Python 3.5
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
from utils import *
from Unet2d_pytorch import UNet
from Unet3d_pytorch import UNet3D
from nnBuildUnits import CrossEntropy3d
import time
import nrrd   # install at https://pypi.org/project/pynrrd/

# Training settings
parser = argparse.ArgumentParser(description="PyTorch InfantSeg")
parser.add_argument("--how2normalize", type=int, default=0, help="how to normalize the data")
parser.add_argument("--batchSize", type=int, default=10, help="training batch size")
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

def main():    
    
    global opt, model 
    opt = parser.parse_args()
    print(opt)
        
#     prefixModelName = 'Segmentor_wce_lrdcr_1112_'
#     prefixPredictedFN = 'preSub_wce_lrdcr_1112_'
#     showTrainLossEvery = 100
#     lr = 1e-4
#     showTestPerformanceEvery = 5000
#     saveModelEvery = 5000
#     decLREvery = 40000
#     numofIters = 200000
#     how2normalize = 0
    
    
    
    #net=UNet()
    net = UNet3D(in_channel=1, n_classes=2) # kernel_size=3, stride=1, padding=1, bias=False, batchnorm=False
    #net.cuda()
    params = list(net.parameters())
    print('len of params is ')
    print(len(params))
    print('size of params is ')
    print(params[0].size())
    
    
    mu_mr = 0.0138117
    mu_t1 = 272.49
    mu_t2 = 49.42
     
    std_mr = 0.0578914
    std_t1 = 1036.12705933
    std_t2 = 193.835485614
 
    
    optimizer = optim.SGD(net.parameters(),lr=opt.lr)
    #criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss()
#     criterion = nn.NLLLoss2d()
    
    given_weight = torch.FloatTensor([1,8])
    
    criterion_3d = CrossEntropy3d()#weight=given_weight
    criterion_3d = criterion_3d
    #inputs=Variable(torch.randn(1000,1,32,32)) #here should be tensor instead of variable
    #targets=Variable(torch.randn(1000,10,1,1)) #here should be tensor instead of variable
#     trainset=data_utils.TensorDataset(inputs, targets)
#     trainloader = data_utils.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
#     inputs=torch.randn(1000,1,32,32)
#     targets=torch.LongTensor(1000)
    
#    path_test ='/Users/kushaagragoyal/Desktop/Independent Study/AllLabelData/Specimen2501L/Data/'
#    path_patients_h5 = '/Users/kushaagragoyal/Desktop/Independent Study/AllLabelData/Specimen2501L/test'
#    path_patients_h5_test ='/Users/kushaagragoyal/Desktop/Independent Study/AllLabelData/Specimen2501L/test'
    path_test = '/Users/George/Documents/Blevins/CT_Segmentations/CarotidArtery/'
    path_patients_X = '/Users/George/Documents/Blevins/CT_Segmentations/original_CT/'
    path_patients_y = '/Users/George/Documents/Blevins/CT_Segmentations/CarotidArtery/'

#    batch_size=10
#    data_generator = Generator_3D_patches(path_patients_h5,opt.batchSize,inputKey1='dataMR',outputKey='dataSeg')
    data_generator = Generator_3D_patches(path_patients_X, path_patients_y, opt.batchSize) # utils.py

########### We'd better use dataloader to load a lot of data,and we also should train several epoches############### 
########### We'd better use dataloader to load a lot of data,and we also should train several epoches############### 

        # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            net.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    running_loss = 0.0
    start = time.time()
    for iter in range(opt.start_epoch,opt.numofIters+1):
        #print('iter %d'%iter)
        
        inputs,labels = data_generator.__next__()
        
        print('shape is ....',inputs.shape)
        labels = np.squeeze(labels)
        labels = labels.astype(int)
        print(labels.shape)
        inputs = np.transpose(inputs,(0,4,2,3,1))
        labels = np.transpose(labels,(0,3,2,1))
        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)
        #inputs = inputs.cuda()
        #labels = labels.cuda()
        #we should consider different data to train
        
        #wrap them into Variable
        inputs,labels = Variable(inputs),Variable(labels)
        
        
        ## (2) update G network: minimize the L1/L2 loss, maximize the D(G(x))
        
#         print inputs.data.shape
        outputG = net(inputs) #here I am not sure whether we should use twice or not
        net.zero_grad()
        
        
        lossG_G = criterion_3d(outputG,torch.squeeze(labels)) 
                
        lossG_G.backward() #compute gradients
        
        #lossG_D.backward()

        
        #for other losses, we can define the loss function following the pytorch tutorial
        
        optimizer.step() #update network parameters

        #print('loss for generator is %f'%lossG.data[0])
        #print statistics
        running_loss = running_loss + lossG_G.data[0]
#         print 'running_loss is ',running_loss,' type: ',type(running_loss)
        
#         print type(outputD_fake.cpu().data[0].numpy())
        
        if iter%opt.showTrainLossEvery==0: #print every 2000 mini-batches
            print('************************************************')
            print('time now is: ' + time.asctime(time.localtime(time.time())))
#             print 'running loss is ',running_loss
            print('average running loss for generator between iter [%d, %d] is: %.6f'%(iter - 100 + 1,iter,running_loss/100))

            print('lossG_G is %.6f respectively.'%(lossG_G.data[0]))

            print('cost time for iter [%d, %d] is %.2f'%(iter - 100 + 1,iter, time.time()-start))
            print('************************************************')
            running_loss = 0.0
            start = time.time()
        if iter%opt.saveModelEvery==0: #save the model
            torch.save(net.state_dict(), opt.prefixModelName+'%d.pt'%iter)
            print('save model: '+opt.prefixModelName+'%d.pt'%iter)
        if iter%opt.decLREvery==0:
            opt.lr = opt.lr*0.1
            adjust_learning_rate(optimizer, opt.lr)
                
        if iter%opt.showTestPerformanceEvery==0: #test one subject  
            # to test on the validation dataset in the format of h5 
            inputs,labels = data_generator_test.__next__()
            labels = np.squeeze(labels)
            labels = labels.astype(int)
            inputs = torch.from_numpy(inputs)
            labels = torch.from_numpy(labels)
            inputs = np.transpose(inputs,(0,4,2,3,1))
            labels = np.transpose(labels,(0,3,2,1))
            #inputs = inputs.cuda()
            #labels = labels.cuda()
            inputs,labels = Variable(inputs),Variable(labels)
            outputG = net(inputs)
            lossG_G = criterion_3d(outputG,torch.squeeze(labels))
            
            print('.......come to validation stage: iter {}'.format(iter),'........')
            print('lossG_G is %.2f.'%(lossG_G.data[0]))


            mr_test_itk=sitk.ReadImage(os.path.join(path_test,'test.nii.gz'))
            ct_test_itk, _ =nrrd.read(os.path.join(path_test,'test.nrrd'))
            
            mrnp=sitk.GetArrayFromImage(mr_test_itk)
    
            ctnp=np.transpose(ct_test_itk,[2,0,1])
            
            ##### specific normalization #####
#             mrnp = (mrnp - mu_mr)/std_mr
            
            mu = np.mean(mrnp)
            
            #for training data in pelvicSeg
            if opt.how2normalize == 1:
                maxV, minV=np.percentile(mrnp, [99 ,1])
                print('maxV,',maxV,' minV, ',minV)
                mrnp=(mrnp-mu)/(maxV-minV)
                print('unique value: ',np.unique(ctnp))

            #for training data in pelvicSeg
            elif opt.how2normalize == 2:
                maxV, minV = np.percentile(mrnp, [99 ,1])
                print('maxV,',maxV,' minV, ',minV)
                mrnp = (mrnp-mu)/(maxV-minV)
                print('unique value: ',np.unique(ctnp))
            
            #for training data in pelvicSegRegH5
            elif opt.how2normalize== 3:
                std = np.std(mrnp)
                mrnp = (mrnp - mu)/std
                print('maxV,',np.ndarray.max(mrnp),' minV, ',np.ndarray.min(mrnp))
                
            elif opt.how2normalize== 4:
                maxV, minV = np.percentile(mrnp, [99.2 ,1])
                print('maxV is: ',np.ndarray.max(mrnp))
                mrnp[np.where(mrnp>maxV)] = maxV
                print('maxV is: ',np.ndarray.max(mrnp))
                mu=np.mean(mrnp)
                std = np.std(mrnp)
                mrnp = (mrnp - mu)/std
                print('maxV,',np.ndarray.max(mrnp),' minV, ',np.ndarray.min(mrnp))

    
#             full image version with average over the overlapping regions
#             ct_estimated = testOneSubject(mrnp,ctnp,[3,168,112],[1,168,112],[1,8,8],netG,'Segmentor_model_%d.pt'%iter)
            
#             sz = mrnp.shape
#             matFA = np.zeros(sz[0],3,sz[2],sz[3],sz[4])
            matFA = mrnp
             #note, matFA and matFAOut same size 
            matGT = ctnp
#                 volFA = sitk.GetImageFromArray(matFA)
#                 sitk.WriteImage(volFA,'volFA'+'.nii.gz')
#                 volGT = sitk.GetImageFromArray(matGT)
#                 sitk.WriteImage(volGT,'volGT'+'.nii.gz')
#             print 'matFA shape: ',matFA.shape
            matOut,_ = testOneSubject(matFA,matGT,10,[16,64,64],[16,64,64],[8,20,20],net,opt.prefixModelName+'%d.pt'%iter)
            print('matOut shape: ',matOut.shape)
            ct_estimated = matOut

            ct_estimated = np.rint(ct_estimated) 
#             ct_estimated = denoiseImg(ct_estimated, kernel=np.ones((20,20,20)))   
            diceBladder = dice(ct_estimated,ctnp,1)
#             diceProstate = dice(ct_estimated,ctnp,2)
#             diceRectumm = dice(ct_estimated,ctnp,3)
            
            print('pred: ',ct_estimated.dtype, ' shape: ',ct_estimated.shape)
            print('gt: ',ctnp.dtype,' shape: ',ct_estimated.shape)
            print('dice1 = ',diceBladder)
#             print 'dice1 = ',diceBladder,' dice2= ',diceProstate,' dice3= ',diceRectumm
            volout = sitk.GetImageFromArray(ct_estimated)
            sitk.WriteImage(volout,opt.prefixPredictedFN+'{}'.format(iter)+'.nii.gz')    
#             netG.save_state_dict('Segmentor_model_%d.pt'%iter)
#             netD.save_state_dic('Discriminator_model_%d.pt'%iter)
        
    print('Finished Training')
    
if __name__ == '__main__':     
    main()
    
