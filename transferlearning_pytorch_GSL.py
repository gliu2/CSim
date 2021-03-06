# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:55:06 2019

Transfer learning tutorial:
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

@author: CTLab
George S. Liu
5-20-19
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
import time
import os
import copy
import random
import math
import dataloading_arriGSL
import densenet_av

plt.ion()   # interactive mode

## Paths to data and mask folders
#PATH_DATA = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/'
#PATH_TRAIN = os.path.join(PATH_DATA, 'train/')
#PATH_VAL = os.path.join(PATH_DATA, 'val/')
#PATH_MASK = os.path.join(PATH_DATA, 'masks/')
#PATH_CSV = os.path.join(PATH_DATA, 'arritissue_sessions.csv')
#PATH_CSV_VAL = os.path.join(PATH_DATA, 'arritissue_sessions_val.csv')

#myclasses = ["Artery",
#"Bone",
#"Cartilage",
#"Dura",
#"Fascia",
#"Fat",
#"Muscle",
#"Nerve",
#"Parotid",
#"PerichondriumWCartilage",
#"Skin",
#"Vein"]
#myclasses = [
#"Fascia",
#"Muscle"]
#num_classes = len(myclasses)

#%% Data loading
#We will use torchvision and torch.utils.data packages for loading the data.
#
#The problem we’re going to solve today is to train a model to classify ants and bees. We have about 120 training images each for ants and bees. There are 75 validation images for each class. Usually, this is a very small dataset to generalize upon, if trained from scratch. Since we are using transfer learning, we should be able to generalize reasonably well.
#
#This dataset is a very small subset of imagenet. 
# Dataset: https://download.pytorch.org/tutorial/hymenoptera_data.zip

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': dataloading_arriGSL.ComposedTransform([
        dataloading_arriGSL.RandomResizeSegmentation(),
        dataloading_arriGSL.RandomCropInSegmentation(32),
        dataloading_arriGSL.ToTensor(),
#        dataloading_arriGSL.ImageStandardizePerImage()
        dataloading_arriGSL.ImageStandardizePerDataset()
    ]),
    'val': dataloading_arriGSL.ComposedTransform([
        dataloading_arriGSL.RandomCropInSegmentation(32),
        dataloading_arriGSL.ToTensor(),
#        dataloading_arriGSL.ImageStandardizePerImage()
        dataloading_arriGSL.ImageStandardizePerDataset()
    ]),
}


# Datasets
transformed_dataset_train = dataloading_arriGSL.ArriTissueDataset(csv_file=PATH_CSV, root_dir=PATH_TRAIN, mask_dir=PATH_MASK,
                                           transform=data_transforms['train'])

transformed_dataset_val = dataloading_arriGSL.ArriTissueDataset(csv_file=PATH_CSV_VAL, root_dir=PATH_VAL, mask_dir=PATH_MASK,
                                           transform=data_transforms['val'])
    
dataloader_train = torch.utils.data.DataLoader(transformed_dataset_train, batch_size=8,
                        shuffle=True, num_workers=4)

dataloader_val = torch.utils.data.DataLoader(transformed_dataset_val, batch_size=8,
                        shuffle=False, num_workers=4)
    
dataloaders = {'train': dataloader_train, 'val': dataloader_val}

dataset_sizes_train = len(transformed_dataset_train)
dataset_sizes_val = len(transformed_dataset_val)
dataset_sizes = {'train': dataset_sizes_train, 'val': dataset_sizes_val}

class_names = myclasses

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#%% Visualize a few images
#Let’s visualize a few training images so as to understand the data augmentations.
def imshow(inp, title=None):
    """Imshow for Tensor."""
    # Undo per-image normalization
    print("Input image is of type {}".format(inp.dtype))
    inp = dataloading_arriGSL.transform_denormalizeperdataset(inp)
    inp = inp.numpy().transpose((1,2,0)) # torch tensor -> numpy
    print("Shown image is of type {}, shape {}".format(inp.dtype, np.shape(inp)))
    
#    # Apply per-image normalization to verify normalization transform accuracy
#    inp = inp.numpy().transpose((1, 2, 0))
#    mean = np.array(np.mean(inp, axis=(0,1)))
#    std = np.array(np.std(inp, axis=(0,1)))
#    inp = (inp - mean)/std
#    inp = np.clip(inp, 0, 1)
    
    # Undo per-batch normalization
#    inp = inp.numpy().transpose((1, 2, 0))
#    mean = MEAN_CHANNEL_PIXELVALS
#    std = STD_CHANNEL_PIXELVALS
#    inp = std * inp + mean
#    inp = np.clip(inp, 0, 1)
    
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
#inputs, classes = next(iter(dataloaders['train']))
#inputs, classes = next(iter(dataloader_train))
sample_batch = next(iter(dataloader_train))
inputs, classes = sample_batch['image'], sample_batch['tissue']

# Make a grid from batch
out = torchvision.utils.make_grid(inputs[:, :3, :, :])

imshow(out, title=[class_names[x] for x in classes])
#imshow(out, title=[x for x in classes])


#%% Train the model
#Now, let’s write a general function to train a model. Here, we will illustrate:
#
#Scheduling the learning rate
#Saving the best model
#In the following, parameter scheduler is an LR scheduler object from torch.optim.lr_scheduler.
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, verbose=True):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    cache_loss = {'train': [], 'val': []}
    cache_acc = {'train': [], 'val': []}
    
    for epoch in range(num_epochs):
        if verbose:
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
#            for inputs, labels in dataloaders[phase]:
#                inputs = inputs.to(device)
#                labels = labels.to(device)
            for ii, sample_batch in enumerate(dataloaders[phase]):
                inputs, labels = sample_batch['image'], sample_batch['tissue']
                inputs = inputs.type(torch.cuda.FloatTensor)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
#                    outputs = model(inputs)
                    outputs = model(inputs[:,:3,:,:]) # RGB train only
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                print(ii)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if verbose:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            # Save losses to plot learning curve
            cache_loss[phase].append((epoch, epoch_loss))
            cache_acc[phase].append((epoch, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        if verbose:
            print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, cache_loss, cache_acc

#%% Visualize model predictions
#    Generic function to display predictions for a few images
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
#        for i, (inputs, labels) in enumerate(dataloaders['val']):
#            inputs = inputs.to(device)
#            labels = labels.to(device)
        for sample_batch in dataloaders['val']:
            inputs_orig, labels = sample_batch['image'], sample_batch['tissue']
            inputs = inputs_orig.type(torch.cuda.FloatTensor)
            inputs = inputs.to(device)
            labels = labels.to(device)

#            outputs = model(inputs)
            outputs = model(inputs[:, :3, :, :]) # RGB image only
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}, true: {}'.format(class_names[preds[j]], class_names[labels[j]]))
#                imshow(inputs.cpu().data[j])
#                out = inputs[:, :3, :, :].cpu().data[j].numpy()
#                imshow((out * 255).astype(np.uint8))
#                imshow(out)
                imshow(inputs_orig.data[j, :3, :, :])
#                print('Visualizing image {} max val {}, {}, {} and min val {}, {}, {}'.format(j, 
#                        torch.max(inputs_orig.data[j, 0, :, :]), torch.max(inputs_orig.data[j, 1, :, :]), torch.max(inputs_orig.data[j, 2, :, :]),
#                        torch.min(inputs_orig.data[j, 0, :, :]), torch.min(inputs_orig.data[j, 1, :, :]), torch.min(inputs_orig.data[j, 2, :, :])))

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        
#%% 6-11-19: Plot Learning curve
def learning_curve(cache_loss, cache_acc):
    # Losses
    randomguess_loss = np.log(len(class_names))
    for phase in ['train', 'val']:
        t_size, losses = zip(*cache_loss[phase])
        plt.plot(t_size, losses)
    plt.hlines(randomguess_loss, xmin=0, xmax=NUM_EPOCHS, colors=(0,1,0), linestyles='dashed') # random-guess loss marked as horizontal green dashed line
    #plt.axis([0, NUM_EPOCHS, 0, 1])
    plt.title('Learning curve, classes=' + str(class_names))
    plt.xlabel('Epoch #')
    plt.ylabel('Cross-entropy loss')
    plt.legend(['train', 'val', 'random guess'])
    #plt.ylim(top=3) 
    plt.show()
    
    # Acc
    randomguess_acc = 1/len(class_names)
    for phase in ['train', 'val']:
        t_size, acc = zip(*cache_acc[phase])
        plt.plot(t_size, acc)
    plt.hlines(randomguess_acc, xmin=0, xmax=NUM_EPOCHS, colors=(0,1,0), linestyles='dashed') # random-guess loss marked as horizontal green dashed line
    plt.title('Learning curve, classes=' + str(class_names))
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.legend(['train', 'val', 'random guess'])
    plt.show()

        
#%%Finetuning the convnet
#Load a pretrained model and reset final fully connected layer.
#model_ft = models.resnet18(pretrained=True)
model_ft = densenet_av.densenet_40_12_bc(pretrained=True)
num_ftrs = model_ft.fc.in_features
#model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft.fc = nn.Linear(num_ftrs, num_classes)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
#optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = optim.Adam(model_ft.parameters(),  lr=0.0001) # defaulat ADAM lr = 0.001

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

#%% Train and evaluate
#It should take around 15-25 min on CPU. On GPU though, it takes less than a minute.
NUM_EPOCHS = 200

model_ft, cache_loss, cache_acc = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=NUM_EPOCHS, verbose=False)

visualize_model(model_ft)

learning_curve(cache_loss, cache_acc)

#%% 6-18-19: Hyperparameter tuning - Train and evaluate
#It should take around 15-25 min on CPU. On GPU though, it takes less than a minute.
NUM_EPOCHS = 25

TRIALS_HYPERPARAM = 30
alpha = np.zeros((TRIALS_HYPERPARAM, 1)) # learning rates
alpha_cacheloss = []
alpha_cacheacc = []
alpha_model = []
for i in range(TRIALS_HYPERPARAM):
    print('Trial {}/{}'.format(i, TRIALS_HYPERPARAM - 1))
    print('-' * 10)
            
    this_lr = math.pow(10, random.uniform(-4,-1)) # explore LR = [0.0001, 0.1]

    optimizer_ft = optim.Adam(model_ft.parameters(),  lr=0.0001) # defaulat ADAM lr = 0.001
    model_ft, cache_loss, cache_acc = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=NUM_EPOCHS, verbose=False)
    
    best_loss_train = np.min(cache_loss['train'], axis=0)[1]
    best_loss_val = np.min(cache_loss['val'], axis=0)[1]
    best_acc_train = np.max(cache_acc['train'], axis=0)[1]
    best_acc_val = np.max(cache_acc['val'], axis=0)[1]
    print('Learning rate: {:.4f}: '.format(this_lr))
    print('Best train loss: {:.4f}:  Acc: {:.4f}'.format(
            best_loss_train, best_acc_train))
    print('Best val loss: {:.4f}:  Acc: {:.4f}'.format(
            best_loss_val, best_acc_val))
        
#    visualize_model(model_ft)
    
#    learning_curve(cache_loss, cache_acc)
    
    # Cache model, learning curves
    alpha[i] = this_lr
    alpha_cacheloss.append(cache_loss)
    alpha_cacheacc.append(cache_acc)
    alpha_model.append(model_ft)

#%% 6-18-19: Record best losses and accuracies
best_loss_train = np.zeros((TRIALS_HYPERPARAM, 1))
best_loss_val = np.zeros((TRIALS_HYPERPARAM, 1))
best_acc_train = np.zeros((TRIALS_HYPERPARAM, 1))
best_acc_val = np.zeros((TRIALS_HYPERPARAM, 1))
for i in range(TRIALS_HYPERPARAM):
    print('Trial {}/{}'.format(i, TRIALS_HYPERPARAM - 1))
    print('-' * 10)
    print('Learning rate: {:.4f}: '.format(alpha[i][0]))
    
    best_loss_train[i] = np.min(alpha_cacheloss[i]['train'], axis=0)[1]
    best_loss_val[i] = np.min(alpha_cacheloss[i]['val'], axis=0)[1]
    best_acc_train[i] = (np.max(alpha_cacheacc[i]['train'], axis=0)[1]).cpu()
    best_acc_val[i] = (np.max(alpha_cacheacc[i]['val'], axis=0)[1]).cpu()
    
    
    print('Best train loss: {:.4f}:  Acc: {:.4f}'.format(
            best_loss_train[i], best_acc_train[i]))
    print('Best val loss: {:.4f}:  Acc: {:.4f}'.format(
            best_loss_val[i], best_acc_val[i]))

#%% ConvNet as fixed feature extractor
#Here, we need to freeze all the network except the final layer. We need to set requires_grad == False to freeze the parameters so that the gradients are not computed in backward().
model_conv = densenet_av.densenet_40_12_bc(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, num_classes)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
#optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
optimizer_conv = optim.Adam(model_ft.parameters())

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

#%% Train and evaluate
#On CPU this will take about half the time compared to previous scenario. This is expected as gradients don’t need to be computed for most of the network. However, forward does need to be computed.
#model_conv = train_model(model_conv, criterion, optimizer_conv,
#                         exp_lr_scheduler, num_epochs=25)
model_conv, cache_loss2, cache_acc2 = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)

visualize_model(model_conv)

learning_curve(cache_loss2, cache_acc2)

plt.ioff()
plt.show()

