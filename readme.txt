Readme for Arrinet convolutional neural network code to classify Arriscope image patches of tissue
George Liu
Last update: 3/26/2020

== Main code files ==
ARRI .raw unprocessed images 
-tile_data_ariraw_GSL.py
-train_arrinet_arrirawTiles.py

Processed TIFF images
-train_arrinetRGB.py - run to train / evaluate neural network
-tile_data_GSL.py

== Dependencies ==
-dataloading_arriGSL.py - contains dataset class for loading batches of Arriscope image patches
-mat.py
-densenet_av.py
-plot_confusion_matrix.py

== Installation ==
Anaconda, Python 3.6, PyTorch
Matlab 2019

More details to be provided later....



