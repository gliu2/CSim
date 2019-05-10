# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:00:17 2019

@author: CTLab

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

Additional code for hyperparameter tuning with cross-validation : https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d

Last edit: 5-9-19
George Liu
Dependencies: arriANN.py (load data)
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, auc
import arriANN
import pickle
import time

#%% Load data
print('Load training data...')
x_train, y_train = arriANN.load_trainingdata()

print('Load validation data...')
x_test, y_test = arriANN.load_validationdata()

# Compress training features, in order:
#- Arriwhite (R, G, B)
#- Blue (B)
#- Green (G)
#- IR (R)
#- Red (R)
# NO VIOLET
#- White (R, G, B)
#
# [1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
channel_isuseful = [1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
indx_usefulchannels = np.nonzero(channel_isuseful)
indx_usefulchannels = np.reshape(indx_usefulchannels, (np.size(indx_usefulchannels),)) # change to rank-1 numpy array

x_train = x_train[:, indx_usefulchannels]
x_test = x_test[:, indx_usefulchannels]


#%% 
n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
#n_estimators = [1, 2, 4, 8]
train_results = []
test_results = []
for estimator in n_estimators:
    print('Working on estimator: ', estimator, ' out of max ', n_estimators[-1])
    start = time.time()
    rf = RandomForestClassifier(n_estimators=estimator, max_depth=6,  n_jobs=-1)
    rf.fit(x_train, y_train)
    train_pred = rf.predict(x_train)
    train_acc = arriANN.accuracy(y_train.numpy(), train_pred)
    train_results.append(train_acc)
    test_pred = rf.predict(x_test)
    test_acc = arriANN.accuracy(y_test.numpy(), test_pred)
    test_results.append(test_acc)
    
    end = time.time()
    print('  Time: ', end - start) # seconds
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_estimators, train_results, 'b', label="Train acc")
line2, = plt.plot(n_estimators, test_results, 'r', label="Test acc")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Accuracy')
plt.xlabel('n_estimators')
plt.show()

##%% Build random forest classifier
#print('Build classifier...')
#clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
#
##%% Fit classifier
#print('Fit classifier...')
#clf.fit(xx, yy)
#
##%% Save random forest classifier
#with open('clf_randforest_arritrain.pkl', 'wb') as ff:
#    pickle.dump(clf, ff)
#    
    
    