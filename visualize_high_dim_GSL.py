# -*- coding: utf-8 -*-
"""
Created on Mon May 13 20:42:07 2019

Code modified from: https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b

@author: CTLab

5-13-19
George Liu
"""

from __future__ import print_function
#%matplotlib inline
import time
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import arriANN


#%% Load data
print('Load training data...')
x_train, y = arriANN.load_trainingdata()

#print('Load validation data...')
#x_test, y_test = arriANN.load_validationdata()

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

X = x_train
# uncomment next line to use 10 compressed features instead of 21
#X = x_train[:, indx_usefulchannels]
#x_test = x_test[:, indx_usefulchannels]

# Convert Tensor matrices to numpy
X = X.numpy()
y = y.numpy()
print(X.shape, y.shape)

# List of tissue type class names
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
"Perichond",
"Vein"]

#%% Convert matrix and vector to Panda Dataframe
#feat_cols = ['arriwhite_R', 'arriwhite_G', 'arriwhite_B', 'blue_B', 'green_G', 'IR_R', 'red_R', 'white_R', 'white_G', 'white_B']
feat_cols = [ str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X,columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))
X, y = None, None
print('Size of the dataframe: {}'.format(df.shape))

#%% Ensure randomization
# For reproducability of the results
#np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])

#%% Visualize MNIST images
#plt.gray()
#fig = plt.figure( figsize=(16,7) )
#for i in range(0,15):
#    ax = fig.add_subplot(3,5,i+1, title="Digit: {}".format(str(df.loc[rndperm[i],'label'])) )
#    ax.matshow(df.loc[rndperm[i],feat_cols].values.reshape((28,28)).astype(float))
#plt.show()

#%% Generate first 3 principal components
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)
df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

#%% Visualize first 2 components
subsample = 100000
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 12),
    data=df.loc[rndperm[:subsample],:],
    legend="full",
    alpha=0.3
)
L=plt.legend()
L.get_texts()[0].set_text('Tissue')
for i in range(12):
    L.get_texts()[i+1].set_text(classes[i])

#%% 3-D version of same plot 
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=df.loc[rndperm[:subsample],:]["pca-one"], 
    ys=df.loc[rndperm[:subsample],:]["pca-two"], 
    zs=df.loc[rndperm[:subsample],:]["pca-three"], 
    c=df.loc[rndperm[:subsample],:]["y"], 
    cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()

#%% T-Sne
#|  It is highly recommended to use another dimensionality reduction
#|  method (e.g. PCA for dense data or TruncatedSVD for sparse data)
#|  to reduce the number of dimensions to a reasonable amount (e.g. 50)
#|  if the number of features is very high.

# Run t-sne on first 10,000 samples with all (784 MNIST) dimension
N = 100000
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset[feat_cols].values
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_subset)
df_subset['pca-one'] = pca_result[:,0]
df_subset['pca-two'] = pca_result[:,1] 
df_subset['pca-three'] = pca_result[:,2]
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

#%% Visualize 2 resulting dimensions
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 12),
    data=df_subset,
    legend="full",
    alpha=0.3
)
L=plt.legend()
L.get_texts()[0].set_text('Tissue')
for i in range(12):
    L.get_texts()[i+1].set_text(classes[i])
    

# Just to compare PCA and t-SNE
plt.figure(figsize=(16,7))
ax1 = plt.subplot(1, 2, 1)
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 12),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax1
)
L=plt.legend()
L.get_texts()[0].set_text('Tissue')
for i in range(12):
    L.get_texts()[i+1].set_text(classes[i])
    
ax2 = plt.subplot(1, 2, 2)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 12),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax2
)
L=plt.legend()
L.get_texts()[0].set_text('Tissue')
for i in range(12):
    L.get_texts()[i+1].set_text(classes[i])

#%% Run PCA to reduce to 50 dimensions
pca_50 = PCA(n_components=10)
pca_result_50 = pca_50.fit_transform(data_subset)
print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

# t-SNE on PCA-reduced data
time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(pca_result_50)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

# And visualization
df_subset['tsne-pca50-one'] = tsne_pca_results[:,0]
df_subset['tsne-pca50-two'] = tsne_pca_results[:,1]
plt.figure(figsize=(16,4))
ax1 = plt.subplot(1, 3, 1)
sns.scatterplot(
    x="pca-one", y="pca-two",
    hue="y",
    palette=sns.color_palette("hls", 12),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax1
)
L=plt.legend()
L.get_texts()[0].set_text('Tissue')
for i in range(12):
    L.get_texts()[i+1].set_text(classes[i])
    
ax2 = plt.subplot(1, 3, 2)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 12),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax2
)
L=plt.legend()
L.get_texts()[0].set_text('Tissue')
for i in range(12):
    L.get_texts()[i+1].set_text(classes[i])

ax3 = plt.subplot(1, 3, 3)
sns.scatterplot(
    x="tsne-pca50-one", y="tsne-pca50-two",
    hue="y",
    palette=sns.color_palette("hls", 12),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax3
)

#%% Visualize 2 resulting dimensions - HIGH RESOLUTION for art of science competition
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
my_dpi = 96
#plt.figure(figsize=(16,10))
plt.figure(figsize=(20, 20), dpi=my_dpi)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 12),
    data=df_subset,
#    legend="full",
    legend=False,
    alpha=0.3
)
plt.xlabel('')
plt.ylabel('')
plt.savefig('my_fig.png', dpi=my_dpi*2)
#L=plt.legend()
#L.get_texts()[0].set_text('Tissue')
#for i in range(12):
#    L.get_texts()[i+1].set_text(classes[i])