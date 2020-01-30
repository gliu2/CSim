# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:00:39 2019

Train ARRInet CNN using tiled 32x32 patches of demosaiced, unprocessed ARRIScope ariraw (.ari) images.

Unprocessed images (21 channels) should be demosaiced by arriRead.m (.ari -> .mat), and 
imported into Python and broken into 32x32 patches by tile_data_ariraw_GSL.py (.mat -> pickle files). 

Input: 
    -unprocessed image data, ndarrays shape (1080, 1920, 21) [pickle format]
    -mask data, ndarrays shape (1080, 1920, ?) [TIFF]
    -CSV file

Code loosely based on Transfer learning tutorial:
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

For plot confusion matrix:
-> First run "%matplotlib qt" command in IPython console to plot confusion matrices in new windows

@author: CTLab
George S. Liu
1-29-20

Dependencies: mat.py, densenet_av.py, plot_confusion_matrix.py
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import os
import copy
import pickle
import mat
import random
import densenet_av
import sys
import pycm

plt.ion()   # interactive mode

MEAN_CHANNEL_PIXELVALS = np.array([
178.3673522,
159.2136305,
145.7320306,
4.50154166,
1.35834203,
26.66413484,
1.10656146,
56.94770658,
5.14473639,
128.0806568,
57.32424962,
93.45134808,
45.17744126,
2.10083132,
3.78928152,
4.84619569,
4.0540281,
4.69607718,
123.0301386,
84.93495828,
25.96622659
])

STD_CHANNEL_PIXELVALS = np.array([
18.05371603,
21.19270681,
24.26668387,
4.87049387,
1.60681874,
12.31887814,
7.65738986,
19.56593939,
10.00423988,
16.50242333,
16.1129809,
15.57760365,
11.38605095,
2.18528431,
4.13749091,
3.18178838,
1.68858641,
2.811973,
22.58913003,
23.27044106,
23.48049806
])
    
# Tile dimensions (don't change)
TILE_HEIGHT = 32
TILE_WIDTH = 32
    
# Paths to data
PATH_PATCHES2D = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/patches2d'
#PATH_PATCHES2D = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/patches2d_sample5'
PATH_PATCHES3D = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/patches3d'
#PATH_PATCHES3D = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/patches3d_sample5'
PATH_OUTPUT2D = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/output2d'
PATH_OUTPUT3D = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/output3d'
#PATH_PATCHES2D = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/patches2d_parotid'
#PATH_PATCHES3D = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/patches3d_parotid'
#PATH_OUTPUT2D = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/output2d_parotid'
#PATH_OUTPUT3D = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/output3d_parotid'
    
# Figure saving parameters
FIG_HEIGHT = 16 # inches
FIG_WIDTH = 12 # inches
FIG_DPI = 200

# Hyper-parameters
LOADMODEL = True
ISMULTISPECTRAL = True
DROPOUT_RATE = 0.0 # densenet paper uses 0.2
ALPHA_L2REG = 0.001 # 1e-5
CM_NORMALIZED = True # confusion matrix normalized?
BATCH_SIZE = 128 # Dunnmon recommends 64-256 (>=16) 
NUM_EPOCHS = 40
LEARNING_RATE = 0.001
LRDECAY_STEP = 10
LRDECAY_GAMMA = 0.1
ISPRETRAINED = False
    
#%% Utilities
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Make input 3-channel RGB if it is multispectral
    if np.shape(inp)[2] > 3:
        inp = inp[:, :, :3]
    mean = np.array(MEAN_CHANNEL_PIXELVALS[:3]) 
    std = np.array(STD_CHANNEL_PIXELVALS[:3])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
        
#    Train the model
#Now, let’s write a general function to train a model. Here, we will illustrate:
#
#Scheduling the learning rate
#Saving the best model
#In the following, parameter scheduler is an LR scheduler object from torch.optim.lr_scheduler.
def train_model(model, criterion, optimizer, scheduler, dataloaders, device, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    cache_loss = {'train': [], 'val': []}
    cache_acc = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Save losses to plot learning curve
            cache_loss[phase].append((epoch, epoch_loss))
            cache_acc[phase].append((epoch, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, cache_loss, cache_acc

#% Visualize model predictions
#    Generic function to display predictions for a few images
def visualize_model(model, dataloaders, device, class_names, num_images=6, columns=2, phase='val'):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    fig.set_canvas(plt.gcf().canvas)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = fig.add_subplot(num_images//columns, columns, images_so_far)
                ax.axis('off')
#                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                ax.set_title('pred: {}, true: {}'.format(class_names[preds[j]], class_names[labels[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        
    return fig

#% 6-11-19: Plot Learning curve
def learning_curve(cache_loss, cache_acc, class_names, num_epochs=25):
    # Losses
    fig1 = plt.figure()
    randomguess_loss = np.log(len(class_names))
    for phase in ['train', 'val']:
        t_size, losses = zip(*cache_loss[phase])
        plt.plot(t_size, losses)
    plt.hlines(randomguess_loss, xmin=0, xmax=num_epochs, colors=(0,1,0), linestyles='dashed') # random-guess loss marked as horizontal green dashed line
    #plt.axis([0, NUM_EPOCHS, 0, 1])
    plt.title('Learning curve, classes=' + str(class_names))
    plt.xlabel('Epoch #')
    plt.ylabel('Cross-entropy loss')
    plt.legend(['train', 'val', 'random guess'])
    #plt.ylim(top=3) 
    plt.show()
    
    # Acc
    fig2 = plt.figure()
    randomguess_acc = 1/len(class_names)
    for phase in ['train', 'val']:
        t_size, acc = zip(*cache_acc[phase])
        plt.plot(t_size, acc)
    plt.hlines(randomguess_acc, xmin=0, xmax=num_epochs, colors=(0,1,0), linestyles='dashed') # random-guess loss marked as horizontal green dashed line
    plt.title('Learning curve, classes=' + str(class_names))
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'val', 'random guess'])
    plt.show()
    
    return fig1, fig2

def pickle_loader(input_file):
    item = pickle.load(open(input_file, 'rb'))
    return item

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
#                          cmap=plt.cm.Blues): 
                          cmap=plt.cm.BuGn):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # hack to make compatible with pytorch tensor confusion matrix
    if torch.is_tensor(cm):
        cm = cm.cpu().numpy()
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
            
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                # original code in for loop
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
            else:
                ax.text(j, i, int(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

#%% Main routine
def main():
  
    # Get start time for this run
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print(timestr)
    
#    for tt in range(30):
#        # Get start time for this run
#        timestr = time.strftime("%Y%m%d-%H%M%S")
#        print(timestr)
#    
#        xx = 3 + random.random()*1
#        LEARNING_RATE = 10**-xx
#        
#        # random search hyperparameters
#        yy = 2 + random.random()*1.5
#        ALPHA_L2REG = 1*10**-yy
#        
#        zz = random.random()*0.1
#        DROPOUT_RATE = zz
#    
#        print('Iteration: ', tt, LEARNING_RATE , ALPHA_L2REG, DROPOUT_RATE)
    
#    for xx in [True, False]:
    for xx in [True]:
        ISMULTISPECTRAL = xx
    
        #%% Data loading
        if ISMULTISPECTRAL:
            # set paths to RGB images
            data_dir = PATH_PATCHES3D
            out_path = PATH_OUTPUT3D
            
            # Data augmentation and normalization for training
            # Just normalization for validation
            data_transforms = {
                'train': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN_CHANNEL_PIXELVALS, STD_CHANNEL_PIXELVALS)
                ]),
                'val': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN_CHANNEL_PIXELVALS, STD_CHANNEL_PIXELVALS)
                ]),
                'test': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN_CHANNEL_PIXELVALS, STD_CHANNEL_PIXELVALS)
                ]),
            }
            
            image_datasets = {x: datasets.DatasetFolder(os.path.join(data_dir, x), loader=pickle_loader, 
                                                        extensions='.pkl', transform=data_transforms[x])
                              for x in ['train', 'val', 'test']}
            num_channels = 21
            
        else: # RGB 3-channel TIFF tiled images
            # set paths to RGB images
            data_dir = PATH_PATCHES2D
            out_path = PATH_OUTPUT2D
                    
            # Data augmentation and normalization for training
            # Just normalization for validation
            data_transforms = {
                'train': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN_CHANNEL_PIXELVALS[:3], STD_CHANNEL_PIXELVALS[:3])
                ]),
                'val': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN_CHANNEL_PIXELVALS[:3], STD_CHANNEL_PIXELVALS[:3])
                ]),
                'test': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN_CHANNEL_PIXELVALS[:3], STD_CHANNEL_PIXELVALS[:3])
                ]),
            }
            
            image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                      data_transforms[x])
                              for x in ['train', 'val', 'test']}
            num_channels = 3
            
        print('Num channels', num_channels)
        if not LOADMODEL:
            dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                     shuffle=True, num_workers=0)
                          for x in ['train', 'val', 'test']}
        else:
            dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                     shuffle=False, num_workers=0)
                          for x in ['train', 'val', 'test']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
        class_names = image_datasets['train'].classes
        num_classes = len(class_names)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(dataset_sizes)
        print(num_classes)
        print(device)
                
        #%%Finetuning the convnet
        #Load a pretrained model and reset final fully connected layer.
    #    model_ft = models.resnet18(pretrained=True)
        model_ft = densenet_av.densenet_40_12_bc(pretrained=ISPRETRAINED, in_channels=num_channels, drop_rate=DROPOUT_RATE)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        
        model_ft = model_ft.to(device)
        
        criterion = nn.CrossEntropyLoss()
        
        # Observe that all parameters are being optimized
    #    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        optimizer_ft = optim.Adam(model_ft.parameters(),  lr=LEARNING_RATE, weight_decay=ALPHA_L2REG) # defaulat ADAM lr = 0.001
        
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=LRDECAY_STEP, gamma=LRDECAY_GAMMA)
#            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1000, gamma=LRDECAY_GAMMA) # For Adam optimizer, no need for LR decay
        
        #%% Train and evaluate
        if LOADMODEL:
            print('Loading model... Select loss/acc file:')
            filepath = mat.uigetfile()
            [cache_loss, cache_acc] = pickle.load(open(filepath, "rb"))
            print('Loading model... ')
            modelpath = filepath.replace('lossacc', 'modelparam').replace('.pkl', '.pt')
            model_ft.load_state_dict(torch.load(modelpath))
            model_ft.eval()
            
            # Get same filename for saving
            path_head, path_tail = os.path.split(filepath)
            filename_pre, path_ext = os.path.splitext(path_tail)
            
#            # 8-25-2019
#            # Obtain per-image classification accuracy based on patches - loop through folders without dataloader
#            for phase in ['train', 'val', 'test']:
#                data_dir2 = os.path.join(data_dir, phase)
#                tissues = os.listdir(data_dir2) # should be num_classes # of folders
#                print('Evaluating per-specimen accuracy on dataset: ', phase)
#                
#                # Iterate over tissue classes
#                for tt, tissue in enumerate(tissues):
#                    tissue_folder = os.path.join(data_dir2, tissue)
#                    tissue_files = os.listdir(tissue_folder)
#                    tissue_dates = [i.split('_', 1)[0] for i in tissue_files]
#                    unique_dates = list(set(tissue_dates))
##                    print(unique_dates)
#                    num_dates = np.size(unique_dates)
#                    
#                    num_patches_tissue_date = np.zeros((num_dates, 1))
#                    num_correctpatches_tissue_date = np.zeros((num_dates, 1))
#                    iscorrect_tissue_date = np.zeros((num_dates, 1))
#                    
#                    # Calculate fraction of correct patch predictions per tissue-date specimen
#                    num_patches = 0
#                    for i, session in enumerate(unique_dates):
##                        print(session)
#                        num_patches_tissue_date[i] = tissue_dates.count(session)
#                        tissue_patches_session_filenames = [item for item in tissue_files if item.startswith(session)]
#                        
#                        # Load patches into one batch of shape [M, C, H, W]
#                        # where M is batch size (# patches), C is # channels
#                        patches_session = np.zeros((int(num_patches_tissue_date[i]), num_channels, TILE_HEIGHT, TILE_WIDTH))
#                        for j, patch_filename in enumerate(tissue_patches_session_filenames):
#                            if ISMULTISPECTRAL:
#                                this_image = pickle_loader(os.path.join(tissue_folder, patch_filename)) # read image, shape (H, W, 21)
#                                mean = np.array(MEAN_CHANNEL_PIXELVALS) 
#                                std = np.array(STD_CHANNEL_PIXELVALS)
#                                inp = (this_image - mean)/std
#                            else:
#                                this_image = mpimg.imread(os.path.join(tissue_folder, patch_filename)) # read image, shape (H, W, 3)
#                                mean = np.array(MEAN_CHANNEL_PIXELVALS[:3]) 
#                                std = np.array(STD_CHANNEL_PIXELVALS[:3])
#                                inp = (this_image - mean)/std
#                                
##                            plt.figure(), plt.imshow(this_image[:,:,:3])
##                            print(os.path.join(tissue_folder, patch_filename))
##                            sys.exit()
#                            patches_session[j] = inp.transpose((2, 0, 1))
#                        
#                        # Predict on patches
#                        with torch.no_grad():
#                            inputs = torch.tensor(patches_session, dtype=torch.float).to(device)
#                            outputs = model_ft(inputs)
#                            _, preds = torch.max(outputs, 1)
#                            
##                        print(preds)
#                        
#                        # Calculate number correct patches
#                        true_label = tt
#                        num_correctpatches_tissue_date[i] = np.sum(preds.cpu().numpy()==true_label)
#                        iscorrect_tissue_date[i] = (num_correctpatches_tissue_date[i]/num_patches_tissue_date[i])>=0.5 # Assume 50% or greater patches predictions gives the overall specimen prediction
#                        
##                        num_patches = num_patches + num_patches_tissue_date[i]
##                        print('  correct', num_correctpatches_tissue_date[i], num_patches_tissue_date[i], iscorrect_tissue_date[i])
##                    print(num_patches)
#                    
#                    # Output per-specimen results
#                    specimens_correct = np.sum(iscorrect_tissue_date)
#                    print('  ', tissue, ': correct specimens ', specimens_correct, ' out of ', num_dates)
            
        else: # Train model from scratch
            print('Train model...')
            # Train
            #It should take around 15-25 min on CPU. On GPU though, it takes less than a minute.
            model_ft, cache_loss, cache_acc = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                   dataloaders, device, dataset_sizes, num_epochs=NUM_EPOCHS)
               
            # Save loss and acc to disk
            filename_pre = 'nclass' + str(num_classes)
            t_size, val_acc = zip(*cache_acc['val']) # Calculate best val acc
            bestval_acc =  max(val_acc).item()
            
#            filename_pre = timestr + '_nclass' + str(num_classes) + '_pretrain' + str(ISPRETRAINED)+ '_batch' + str(BATCH_SIZE) + '_epoch' + str(NUM_EPOCHS) + '_lr' + str(LEARNING_RATE) + '_' + str(LRDECAY_STEP) + '_' + str(LRDECAY_GAMMA) + '_val' +"{:.4f}".format(bestval_acc)
            filename_pre = timestr + '_multispec' + str(ISMULTISPECTRAL) + '_nclass' + str(num_classes) + '_pretrain' + str(ISPRETRAINED)+ '_batch' + str(BATCH_SIZE) + '_epoch' + str(NUM_EPOCHS) + '_lr' + str(LEARNING_RATE) + '_L2reg' + str(ALPHA_L2REG) + '_DROPOUT' + str(DROPOUT_RATE) + '_val' +"{:.4f}".format(bestval_acc)
            filename = 'lossacc_' + filename_pre + '.pkl'
            pickle.dump([cache_loss, cache_acc], open(os.path.join(out_path, filename), "wb" ))
            
            # Save trained model's parameters for inference
            filename2 = 'modelparam_' + filename_pre + '.pt'
            torch.save(model_ft.state_dict(), os.path.join(out_path, filename2))
        
        # Evaluate 
        model_ft.eval() # set dropout and batch normalization layers to evaluation mode before running inference
        
#        # Examine each figure and output to get granular prediction output info
##        fig0 = visualize_model(model_ft,  dataloaders, device, class_names, num_images=90, columns=10)
##        fig0 = visualize_model(model_ft,  dataloaders, device, class_names, num_images=10) # visualize validation images
#        fig0 = visualize_model(model_ft,  dataloaders, device, class_names, num_images=288, columns=12, phase='test')
#        # Save visualization figure
#        fig0_filename = 'visualize_' + filename_pre + '.png'
#        fig0 = plt.gcf()
#        fig0.set_size_inches(FIG_HEIGHT, FIG_WIDTH)
#        plt.savefig(os.path.join(out_path, fig0_filename), bbox_inches='tight', dpi=FIG_DPI)
        
        fig1, fig2 = learning_curve(cache_loss, cache_acc, class_names, num_epochs=NUM_EPOCHS)
        
        # Save learning curve figures
        fig1_filename = 'losscurve_' + filename_pre + '.png'
        fig2_filename = 'acccurve_' + filename_pre + '.png'
        fig1.set_size_inches(FIG_HEIGHT, FIG_WIDTH)
        fig2.set_size_inches(FIG_HEIGHT, FIG_WIDTH)
        fig1.savefig(os.path.join(out_path, fig1_filename), bbox_inches='tight', dpi=FIG_DPI)
        fig2.savefig(os.path.join(out_path, fig2_filename), bbox_inches='tight', dpi=FIG_DPI)
        
        # Display confusion matrix
        for phase in ['train', 'val', 'test']:
            confusion_matrix = torch.zeros(num_classes, num_classes)
            y_actu = []
            y_pred = []
            with torch.no_grad():
                for i, (inputs, classes) in enumerate(dataloaders[phase]):
                    inputs = inputs.to(device) # shape [128, 21, 36, 36]
                    classes = classes.to(device)
                    outputs = model_ft(inputs)
                    _, preds = torch.max(outputs, 1)
                    for t, p in zip(classes.view(-1), preds.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
                    
                    # Vector of class labels and predictions
                    y_actu = np.hstack((y_actu, classes.view(-1).cpu().numpy()))
                    y_pred = np.hstack((y_pred, preds.view(-1).cpu().numpy()))
        
    #        print(confusion_matrix)
            print(confusion_matrix.diag()/confusion_matrix.sum(1)) # per-class accuracy
            fig3 = plot_confusion_matrix(confusion_matrix, classes=class_names, normalize=CM_NORMALIZED,
                                  title='Confusion matrix, ' + phase)
            
            # Save confusion matrix figure
            fig3_filename = 'cm' + phase + '_' + filename_pre + '.pdf'
            fig3.set_size_inches(FIG_HEIGHT, FIG_WIDTH)
            fig3.savefig(os.path.join(out_path, fig3_filename), bbox_inches='tight', dpi=FIG_DPI)
        
            # Display confusion matrix analysis
            cm2 = pycm.ConfusionMatrix(actual_vector=y_actu, predict_vector=y_pred) # Create CM From Data
#            cm2 = pycm.ConfusionMatrix(matrix={"Class1": {"Class1": 1, "Class2":2}, "Class2": {"Class1": 0, "Class2": 5}}) # Create CM Directly
            cm2 # line output: pycm.ConfusionMatrix(classes: ['Class1', 'Class2'])
            print(cm2)
            
    #    #%% ConvNet as fixed feature extractor
    #    #Here, we need to freeze all the network except the final layer. We need to set requires_grad == False to freeze the parameters so that the gradients are not computed in backward().
    #    
    #    model_conv = torchvision.models.resnet18(pretrained=True)
    #    for param in model_conv.parameters():
    #        param.requires_grad = False
    #    
    #    # Parameters of newly constructed modules have requires_grad=True by default
    #    num_ftrs = model_conv.fc.in_features
    #    model_conv.fc = nn.Linear(num_ftrs, 2)
    #    
    #    model_conv = model_conv.to(device)
    #    
    #    criterion = nn.CrossEntropyLoss()
    #    
    #    # Observe that only parameters of final layer are being optimized as
    #    # opposed to before.
    #    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    #    
    #    # Decay LR by a factor of 0.1 every 7 epochs
    #    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    #    
    #    #%% Train and evaluate
    #    #On CPU this will take about half the time compared to previous scenario. This is expected as gradients don’t need to be computed for most of the network. However, forward does need to be computed.
    #    model_conv, cache_loss2, cache_acc2 = train_model(model_conv, criterion, optimizer_conv,
    #                             exp_lr_scheduler, dataloaders, device, dataset_sizes, num_epochs=NUM_EPOCHS)
    #    
    #    visualize_model(model_conv, dataloaders, device, class_names)
    #    
    #    learning_curve(cache_loss2, cache_acc2, class_names, num_epochs=NUM_EPOCHS)
        
    plt.ioff()
    plt.show()
    
if __name__=='__main__':
    main()