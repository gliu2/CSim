# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:01:52 2019

Plot learning curve

Run after 'train_csimnet.py'

@author: George Liu
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

# Getting back the objects:
with open('GL_3dunet_train_loss.pkl', 'rb') as ff:  
    cache_loss = pickle.load(ff)
    
# Convert list of tuples to two tuples
t_size, losses = zip(*cache_loss)

# Plot learning curve
plt.plot(t_size, losses)
plt.axis([0, 500, 0, 1])
plt.title('Learning curve')
plt.xlabel('Mini-batch #')
plt.ylabel('Cross-entropy loss')
plt.show()