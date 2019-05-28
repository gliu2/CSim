# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:53:07 2019

DATA LOADING AND PROCESSING TUTORIAL
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

Before running:
    -Update CSV file with latest tissue dataset: arritissue_sessions.csv (C:/Users/CTLab/Documents/George/Python_data/arritissue_data)

@author: CTLab
5-24-19
George Liu
"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from os import listdir
from os.path import isfile, join, isdir
from PIL import Image


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

# Paths to data and mask folders
PATH_DATA = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/'
PATH_TRAIN = os.path.join(PATH_DATA, 'train/')
PATH_VAL = os.path.join(PATH_DATA, 'val/')
PATH_MASK = os.path.join(PATH_DATA, 'masks/')
PATH_CSV = os.path.join(PATH_DATA, 'arritissue_sessions.csv')

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
num_classes = len(classes)

illuminations = ["arriwhite",
"blue",
"green",
"IR",
"red",
"violet",
"white"]
num_lights = len(illuminations) # 7

#%% Let’s quickly check number of training images per tissue type.
sessions_frame = pd.read_csv(os.path.join(PATH_DATA, 'arritissue_sessions.csv'))

train_sessions = [f for f in listdir(PATH_TRAIN) if isdir(join(PATH_TRAIN, f))] # Get only folders
#val_sessions = [f for f in listdir(PATH_VAL) if isdir(join(PATH_VAL, f))]
num_sessions = len(train_sessions)
session_tissues = np.zeros((num_sessions, 1))
for i in range(num_sessions):
    print('Session: {}'.format(train_sessions[i]))
    path_sessioni = os.path.join(PATH_TRAIN, train_sessions[i])
    acquisition_labels = [f for f in listdir(path_sessioni) if isdir(join(path_sessioni, f))] # Get only folders
    print('Tissue: {}'.format(len(acquisition_labels)))
    
    # Check that each tissue acquisition folder contains only 7 images (per illumination) and that they are all TIF
    for j in range(len(acquisition_labels)):
        this_folder = os.path.join(path_sessioni, acquisition_labels[j])
        this_files = listdir(this_folder)
        this_tiffs = [f for f in os.listdir(this_folder) if f.endswith('.tif')]
        n_files = len(this_files)
        n_tiffs = len(this_tiffs)
        assert n_tiffs==n_files, "Session " + str(train_sessions[i]) + " " + str(acquisition_labels[j]) + " folder contains " + str(n_files) + " files of which only " + str(n_tiffs) + " are tiff."
        assert n_tiffs==num_lights, "Number of tissue images " + str(n_tiffs) + " is different from number of illumination lights " + str(num_lights)
#        print('Tissue: {}'.format(acquisition_labels[j]))
        
        # Cache number of tissue specimens per session

#%% Let’s write a simple helper function to show an image and its landmarks and use it to show a sample.

def show_tissue(image, tissue):
    """Show image with tissue title"""
    plt.imshow(image[:,:,:3]) # show white illumination RGB for visualization
    plt.title(tissue)

plt.figure()
trial_image = mpimg.imread(os.path.join(this_folder, this_tiffs[0]))
show_tissue(trial_image, acquisition_labels[0])
plt.show()

#%% Dataset class
#torch.utils.data.Dataset is an abstract class representing a dataset. Your custom dataset should inherit Dataset and override the following methods:
#
#__len__ so that len(dataset) returns the size of the dataset.
#__getitem__ to support the indexing such that dataset[i] can be used to get ith sample
#Let’s create a dataset class for our face landmarks dataset. We will read the csv in __init__ but leave the reading of images to __getitem__. This is memory efficient because all the images are not stored in the memory at once but read as required.
#
#Sample of our dataset will be a dict {'image': image, 'tissue': tissue type (string)}. Our dataset will take an optional argument transform so that any required processing can be applied on the sample. We will see the usefulness of transform in the next section.
    
class ArriTissueDataset(Dataset):
    """Arriscope tissue dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.sessions_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return np.count_nonzero(sessions_frame.iloc[:,1:].values)

    def __getitem__(self, idx):
        # Flat indexing of tissues by acqusition session, skipping missing tissue samples not acquired in some sessions
        istissue = sessions_frame.iloc[:,1:].values
        sub_row, sub_col = np.nonzero(istissue)
#        flat_ind = np.ravel_multi_index(np.nonzero(istissue), np.shape(istissue))
#        this_row, this_col = np.unravel_index(flat_ind[idx])
        this_row = sub_row[idx]
        this_col = sub_col[idx]
        this_session = sessions_frame.iloc[this_row, 0]
        this_tissue = sessions_frame.columns[this_col + 1]
        
        # Read all 7 tiff narrowband image corresponding to session, concatenating in spectral dimension as 21 channel image
        this_folder = os.path.join(self.root_dir, str(this_session), this_tissue)
        this_tiffs = [f for f in os.listdir(this_folder) if f.endswith('.tif')]
        n_tiffs = len(this_tiffs)
        # Check that there are only 7 tiff images (for 7 illumination conditions)
        assert n_tiffs==num_lights, "Number of tissue TIFF images " + str(n_tiffs) + " is different from number of illumination lights " + str(num_lights)
        
        image = np.nan
        for img_name in this_tiffs:
            this_image = mpimg.imread(os.path.join(this_folder, img_name))
            if np.isnan(np.sum(image)): # no concatenation necessary for 1st image
                image = this_image
            else:
                image = np.concatenate((image, this_image), axis = 2)
                
        sample = {'image': image, 'tissue': this_tissue}

        if self.transform:
            # TODO: add crop to mask
            sample = self.transform(sample)

        return sample

#%% instantiate this class and iterate through the data samples. We will print the sizes of first 4 samples and show their landmarks.
tissue_dataset = ArriTissueDataset(csv_file=PATH_CSV, root_dir=PATH_TRAIN)

fig = plt.figure()

for i in range(len(tissue_dataset)):
    sample = tissue_dataset[i]

    print(i, sample['image'].shape, sample['tissue'])

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}: {}'.format(i, sample['tissue']))
    ax.axis('off')
    show_tissue(**sample)

    if i == 3:
        plt.show()
        break
    
#%% Transforms
#One issue we can see from the above is that the samples are not of the same size. Most neural networks expect the images of a fixed size. Therefore, we will need to write some prepocessing code. Let’s create three transforms:
#
#Rescale: to scale the image
#RandomCrop: to crop from image randomly. This is data augmentation.
#ToTensor: to convert the numpy images to torch images (we need to swap axes).
#We will write them as callable classes instead of simple functions so that parameters of the transform need not be passed everytime it’s called. For this, we just need to implement __call__ method and if required, __init__ method. We can then use a transform like this:
#
#tsfm = Transform(params)
#transformed_sample = tsfm(sample)
#Observe below how these transforms had to be applied both on the image and landmarks.
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, tissue = sample['image'], sample['tissue']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}
        
#%% Compose transforms
#Now, we apply the transforms on an sample.
#
#Let’s say we want to rescale the shorter side of the image to 256 and then randomly crop a square of size 224 from it. 
# i.e, we want to compose Rescale and RandomCrop transforms. torchvision.transforms.Compose is a simple callable class which allows us to do this.
scale = Rescale((1080, 1920))
crop = RandomCrop(32)
composed = transforms.Compose([scale, crop])

# Apply each of the above transforms on sample.
fig = plt.figure()
sample = tissue_dataset[63]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_tissue(**transformed_sample)

plt.show()

#%% Iterating through the dataset
#Let’s put this all together to create a dataset with composed transforms. To summarize, every time this dataset is sampled:
#
#An image is read from the file on the fly
#Transforms are applied on the read image
#Since one of the transforms is random, data is augmentated on sampling
#We can iterate over the created dataset with a for i in range loop as before.

transformed_dataset = FaceLandmarksDataset(csv_file=os.path.join(PATH_DATA, 'face_landmarks.csv'),
                                           root_dir=PATH_DATA,
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['landmarks'].size())

    if i == 3:
        break
    
#%% However, we are losing a lot of features by using a simple for loop to iterate over the data. In particular, we are missing out on:
#
#Batching the data
#Shuffling the data
#Load the data in parallel using multiprocessing workers.
#torch.utils.data.DataLoader is an iterator which provides all these features. Parameters used below should be clear. One parameter of interest is collate_fn. You can specify how exactly the samples need to be batched using collate_fn. However, default collate should work fine for most use cases.

def main():
    
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)
    
    
    # Helper function to show a batch
    def show_landmarks_batch(sample_batched):
        """Show image with landmarks for a batch of samples."""
        images_batch, landmarks_batch = \
                sample_batched['image'], sample_batched['landmarks']
        batch_size = len(images_batch)
        im_size = images_batch.size(2)
    
        grid = utils.make_grid(images_batch)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
    
        for i in range(batch_size):
            plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                        landmarks_batch[i, :, 1].numpy(),
                        s=10, marker='.', c='r')
    
            plt.title('Batch from dataloader')
    
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['landmarks'].size())
    
        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break
        
    # your complete code

if __name__=='__main__':
    main()
