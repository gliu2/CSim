Readme for Arrinet convolutional neural network code to classify Arriscope image patches of tissue
George Liu
Last update: 3/26/2020

== Main code files ==
ARRI .raw unprocessed images 
-tile_data_ariraw_GSL.py - run to train / evaluate neural network
-train_arrinet_arrirawTiles.py - convert full images ([1]) to tiled 32x32 pixel images ([2])

Processed TIFF images
-train_arrinetRGB.py - run to train / evaluate neural network
-tile_data_GSL.py - convert full images ([1]) to tiled 36x36 pixel images ([2])

== Dependencies ==
-dataloading_arriGSL.py - contains dataset class for loading batches of Arriscope image patches
-mat.py
-densenet_av.py
-plot_confusion_matrix.py

== Data Files ==
[1] full images - split into 'train', 'val', 'test' folders
[2] tiled or patch images - 

== Installation ==
Anaconda, Python 3.6, PyTorch
Matlab 2019

More details to be provided later....



