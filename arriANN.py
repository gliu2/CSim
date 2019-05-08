# -*- coding: utf-8 -*-
"""
arriANN.py
Train and validate a fully-connected neural network (arriANN) for pixelwise classification of tissue narrowband multispectral images.
Only first 21 RGB sensor features used. (Last 3 RGB sensors are thrown out if contains whitemix17 sensors)

Created on Mon May  6 13:54:13 2019

@author: CTLab

Last edit: 5-8-19
George S. Liu

"""

# -*- coding: utf-8 -*-
import numpy as np
import torch
import pickle
from sklearn.metrics import confusion_matrix
import hdf5storage

#%% Custom functions

# Converts scores output from model to predicted class labels
# Input: out_scores - numpy array or tensor (?) of shape (m, NUM_TISSUES) of arriNET model activation outputs from last (softmax) layer
# Output: y_pred - numpy rank-1 array of label predictions from scores
def scores_to_labels(out_scores):
    y_pred = np.argmax(out_scores, axis=1)
    return y_pred

# Predict on validation set at end
#
# model is trained arrinet model
# m_data is the number of examples in input data
# num_classes is number of unique classes (>=2)
# batch_size is batch size
# x_data is a m float torch tensor of shape (m_data, num_features)
# y_data is a long torch tensor of shape (m_data,) rank-1 numpy array
#
# Returns: y_pred - numpy array
def predict(model, num_classes, batch_size, device, x_data):
    m_data = x_data.size()[0]
    with torch.no_grad():

        y_data_scores = np.zeros((m_data, num_classes))
        # Cycle through mini-batches of data to calculate predictions
        for k in range(0, m_data, batch_size):
            if m_data - k >= batch_size:
                indices_val = np.arange(k, k+batch_size)
            else:
                indices_val = np.arange(k, m_data)
            batch_x_data = x_data[indices_val]
            
            #  pushing tensors to CUDA device if available (you have to reassign them)
            batch_x_data = batch_x_data.to(device)
            
            out = model(batch_x_data)
            y_data_scores[indices_val] = out.cpu()
            
        y_pred = scores_to_labels(y_data_scores)
        return y_pred
        
# Calculate accuracy given true and predicted label vectors
# y_true is numpy rank-1 array
# y_pred is numpy rank-1 array
# NOTE: If both inputs are torch Tensors (on CPU) then this method will cause Kernel to crash   
def accuracy(y_true, y_pred):
        # Compute accuracy of predictions
        num_correct = sum(y_true==y_pred)
        acc = num_correct / np.shape(y_true)[0]
        return acc
    
    
    
#%% Train Arrinet
def main():
    NUM_FEATURES = 21 # number of RGB channel features to use for classifying each pixel

    dtype = torch.float
    #device = torch.device("cpu")
    device = torch.device("cuda:0") # Uncomment this to run on GPU
    
    # Load training data
    print('Load training data...')
    filepath = "C:/Users/CTLab/Documents/George/Arri_analysis_4-29-19/kmeans_data_5-6-19.mat"
    if 'mat2' in locals(): print('Yes')
    mat2 = hdf5storage.loadmat(filepath)    
    x = torch.from_numpy(mat2['X_single'][:,:NUM_FEATURES]) # make 24 -> 21 features as needed
    y = torch.from_numpy(mat2['y_label']).long()-1 # convert to long tensor for loss function
    y = y.squeeze(1)
    
    m_training = x.size()[0]
    num_classes = np.size(np.unique(y))
    classes = ["Artery",
    "Bone",
    "Cartilage",
    "Dura",
    "Fascia",
    "Fat",
    "Muscle",
    "Nerve",
    "Skin",
    "Parotid",
    "PerichondriumWCartilage",
    "Vein"]
    
    #%%Load validation data
    print('Load validation data...')
    filepath_val = "C:/Users/CTLab/Documents/George/Arri_analysis_5-6-19/FreshCadaver004_20190429_data_5-6-19_pyfriendly.mat"
    mat3 = hdf5storage.loadmat(filepath_val)
    x_val = torch.from_numpy(mat3['X'][:,:NUM_FEATURES]).float() # make 24 -> 21 features as needed; convert to float tensor
    y_val = torch.from_numpy(mat3['y_label']).long()-1 # convert to long tensor for loss function
    y_val = y_val.squeeze(1) 
    
    m_validation = x_val.size()[0]
    
    #%% Train network
    print('Initialize model...')
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    D_in, H, D_out = NUM_FEATURES, 100, num_classes
    
    ## Create random Tensors to hold inputs and outputs
    #x = torch.randn(N, D_in)
    #y = torch.randn(N, D_out)
    
    # Use the nn package to define our model and loss function.
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
        torch.nn.Softmax(dim=1) # hidden activations are of shape m x D_out, where output features per training example are in 2nd dimension (along rows)
    )
    model = model.cuda()
    #loss_fn = torch.nn.MSELoss(reduction='sum')
    loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean') # input has to be a Tensor of size either (minibatch, C)(minibatch,C)
    
    # Use the optim package to define an Optimizer that will update the weights of
    # the model for us. Here we will use Adam; the optim package contains many other
    # optimization algoriths. The first argument to the Adam constructor tells the
    # optimizer which Tensors it should update.
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    #for t in range(500):
    #    # Forward pass: compute predicted y by passing x to the model.
    #    y_pred = model(x)
    #
    #    # Compute and print loss.
    #    loss = loss_fn(y_pred, y)
    #    print(t, loss.item())
    #
    #    # Before the backward pass, use the optimizer object to zero all of the
    #    # gradients for the variables it will update (which are the learnable
    #    # weights of the model). This is because by default, gradients are
    #    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    #    # is called. Checkout docs of torch.autograd.backward for more details.
    #    optimizer.zero_grad()
    #
    #    # Backward pass: compute gradient of the loss with respect to model
    #    # parameters
    #    loss.backward()
    #
    #    # Calling the step function on an Optimizer makes an update to its
    #    # parameters
    #    optimizer.step()
    
    n_epochs = 100 # or whatever
    batch_size = 128000 # or whatever
    
    print('Start training model...')
    cache_loss = []
    #  pushing tensors to CUDA device if available (you have to reassign them)
    x = x.to(device)
    y = y.to(device)
        #  pushing tensors to CUDA device if available (you have to reassign them)
    x_val_gpu = x_val.to(device)
    y_val_gpu = y_val.to(device)
    for epoch in range(n_epochs):
        # Store training and validation output scores to calculate accuracy of model after each epoch
        num_train_batches = int(np.ceil(m_training/batch_size))
        num_validation_batches = int(np.ceil(m_validation/batch_size))
        cache_training_acc = np.zeros(num_train_batches) # numpy rank-1 array
        cache_validation_acc = np.zeros(num_validation_batches) # numpy rank-1 array
    
        # x is a torch Variable
        permutation = torch.randperm(m_training)
    
        for i in range(0, m_training, batch_size):
            optimizer.zero_grad()
    
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = x[indices], y[indices]  # includes last mini-batch even if its size < batch_size 
            
#            #  pushing tensors to CUDA device if available (you have to reassign them)
#            batch_x = batch_x.to(device)
#            batch_y = batch_y.to(device)
    
            # Compute and print loss.
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
    #        if np.mod(i/batch_size, 50) == 0:
    #            print('Epoch:', epoch, '  Batch ', i/batch_size, 'out of', np.floor(x.size()[0]/batch_size), '  Loss:',  loss.item())
    
#            # Cache outputs to calculate training accuracy at end of epoch
            pred_training_labels = scores_to_labels(outputs.detach().cpu())
            cache_training_acc[int(i/batch_size)] = accuracy(batch_y.cpu().numpy(), pred_training_labels.numpy())
    
            # After each epoch, save training and validation losses to plot learning curve
            if (m_training - i) <= batch_size: # last mini-batch of epoch
                # empty CUDA cache every epoch
                torch.cuda.empty_cache()
                
                with torch.no_grad():
                    model.eval()
                    
                    # Cycle through mini-batches of validation data to calculate mean loss
                    permutation2 = torch.randperm(m_validation)
                    
                    val_loss = []
                    for j in range(0, m_validation, batch_size):
                        indices_val = permutation2[j:j+batch_size]
                        batch_x_val, batch_y_val = x_val_gpu[indices_val], y_val_gpu[indices_val]
#                        batch_x_val, batch_y_val = x_val[indices_val], y_val[indices_val]
                        
#                        #  pushing tensors to CUDA device if available (you have to reassign them)
#                        batch_x_val_gpu = batch_x_val.to(device)
#                        batch_y_val_gpu = batch_y_val.to(device)
                        
#                        out = model(batch_x_val_gpu)
#                        loss_val = loss_fn(out, batch_y_val_gpu)
                        out = model(batch_x_val)
                        loss_val = loss_fn(out, batch_y_val)
                        val_loss.append(loss_val.item())
                        
                        # Cache outputs to calculate validation accuracy at end of epoch
                        pred_validation_labels = scores_to_labels(out.detach().cpu())
                        cache_validation_acc[int(j/batch_size)] = accuracy(batch_y_val.cpu().numpy(), pred_validation_labels.numpy())
#                        cache_pred_validation[j:j+batch_size] = scores_to_labels(out.cpu().detach().numpy())
                        
                    # Calculate mean loss across validation mini-batches -- **NOTE does not take weighted mean of mini-batches' losses if last mini-batch's size is smaller
                    mean_val_loss = np.mean(val_loss)
                    acc_train = np.mean(cache_training_acc)
                    acc_val = np.mean(cache_validation_acc)
                
#                    # Calculate training and validation accuracy
##                    train_pred = scores_to_labels(cache_out_training)
#                    acc_train = accuracy(y.numpy(), cache_pred_training)
#
##                    val_pred = scores_to_labels(cache_out_validation)
#                    acc_val = accuracy(y_val.numpy(), cache_pred_validation)
                
                    # Print trainng and validation losses
                    print('Epoch:', epoch, '  Batch ', i/batch_size, 'out of', np.floor(m_training/batch_size), '  Loss:',  loss.item(), 'Val loss:', mean_val_loss, '   Train acc:', acc_train, 'Val acc:', acc_val)
#                    print('Epoch:', epoch, '  Batch ', i/batch_size, 'out of', np.floor(m_training/batch_size), '  Loss:',  loss.item())

                
                    # Save losses to plot learning curve
                    cache_loss.append((epoch, loss.item(), mean_val_loss, acc_train, acc_val))
    
                    # reset model to training mode for next epoch
                    model.train()
                
            # For other mini-batches, just print training loss every nth mini-batch
            elif np.mod(i/batch_size, 50) == 0:
                print('Epoch:', epoch, '  Batch ', i/batch_size, 'out of', np.floor(m_training/batch_size), '  Loss:',  loss.item())
    
            loss.backward()
            optimizer.step()
            
    #%% Save cached variables to analyze training / learning curve
    with open('arriANN_train_loss.pkl', 'wb') as ff:
        pickle.dump(cache_loss, ff)
    
    # Save trained model's parameters for inference
    PATH = "C:/Users/CTLab/Documents/George/Arri_analysis_5-6-19/arrinet_ann_model_parameters_5-8-19.pt"
    torch.save(model.state_dict(), PATH)
    
    print('Done')
            
    #%% Predict on validation set at end
    model.eval()
    y_pred = predict(model, num_classes, batch_size, device, x_val)
    y_val_true = y_val.numpy()
    
        
    print('Computing confusion matrix...')
    conf = confusion_matrix(y_val_true, y_pred)


if __name__ == "__main__": 
    main()