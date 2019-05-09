# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:00:17 2019

@author: CTLab

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

Last edit: 5-8-19
George Liu
Dependencies: arriANN.py (load data)
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import arriANN
import pickle

#%% Load data
print('Load training data...')
xx, yy = arriANN.load_trainingdata()

perm

#%% Build random forest classifier
print('Build classifier...')
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

#%% Fit classifier
print('Fit classifier...')
clf.fit(xx, yy)

#%% Save random forest classifier
with open('clf_randforest_arritrain.pkl', 'wb') as ff:
    pickle.dump(clf, ff)