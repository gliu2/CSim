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
import matplotlib.colors as colors
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from os import listdir
from os.path import isfile, join, isdir
from PIL import Image
import time
from operator import itemgetter


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

# Paths to data and mask folders
PATH_DATA = 'C:/Users/CTLab/Documents/George/Python_data/arritissue_data/'
PATH_TRAIN = os.path.join(PATH_DATA, 'train/')
PATH_VAL = os.path.join(PATH_DATA, 'val/')
PATH_MASK = os.path.join(PATH_DATA, 'masks/')
#PATH_CSV = os.path.join(PATH_DATA, 'arritissue_sessions.csv')
#PATH_CSV_VAL = os.path.join(PATH_DATA, 'arritissue_sessions_val.csv')
PATH_CSV = os.path.join(PATH_DATA, 'arritissue_sessions_4class.csv')
PATH_CSV_VAL = os.path.join(PATH_DATA, 'arritissue_sessions_val_4class.csv')

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
myclasses = [
"Bone",
"Fascia",
"Fat",
"Muscle"
]
num_classes = len(myclasses)

illuminations = ["arriwhite",
"blue",
"green",
"IR",
"red",
"violet",
"white"]
num_lights = len(illuminations) # 7

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

# Reshape into rank-2 numpy arrays
#MEAN_CHANNEL_PIXELVALS = np.reshape(MEAN_CHANNEL_PIXELVALS, (np.size(MEAN_CHANNEL_PIXELVALS), 1))
#STD_CHANNEL_PIXELVALS = np.reshape(STD_CHANNEL_PIXELVALS, (np.size(STD_CHANNEL_PIXELVALS), 1))

#%% 6-3-19: Convert string/char tissue class name to categorical integer
def tissueclass_str2int(tissue_name, class_names=myclasses):
    categorical_key = pd.get_dummies(class_names)
    return int(categorical_key[tissue_name].values.argmax(0))

def tissueclass_int2str(tissue_int, class_names=myclasses):
    categorical_key = pd.get_dummies(class_names)
    tissue_col = categorical_key.values.argmax(axis=1)[tissue_int]
    return categorical_key.columns[tissue_col]

#%% Let’s write a simple helper function to show an image and its landmarks and use it to show a sample.

def show_tissue(image, tissue):
    """Show image with tissue title"""
    this_image = image[:,:,:3] # show white illumination RGB for visualization
    
#    # normalize image to [0..255] for integers to avoid imshow clipping data outside range
#    if np.amin(this_image)<0:
#        this_image += np.abs(np.amin(this_image))
#    if np.amax(this_image)>255:
#        this_image /= np.amax(this_image)
    
#    print(this_image.dtype)
#    print('Max per channel: {}'.format(np.amax(this_image, axis=(0,1))))
#    print('Min per channel: {}'.format(np.amin(this_image, axis=(0,1))))
#    plt.imshow(this_image, norm=colors.Normalize(vmin=-1.0, vmax=1.0)) 
    plt.imshow(this_image) 

    if isinstance(tissue, str):
        plt.title(tissue)
    else:
        plt.title(tissueclass_int2str(tissue))
        

#%% 5-30-19:  Crop image to segmentation
# execute get_segment_crop(rgb, mask=segment_mask) while rgb is an ndarray of shape (w,h,c) and segment_mask is a boolean ndarray (i.e. containing True/False entries) of shape (w,h), given that w=width, h=height, c=channel.
# https://stackoverflow.com/questions/40824245/how-to-crop-image-based-on-binary-mask
def get_segment_crop(img,tol=0, mask=None):
    if mask is None:
        mask = img > tol
#    return img[np.ix_(mask.any(1), mask.any(0))], mask[np.ix_(mask.any(1), mask.any(0))]
    return img, mask


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

    def __init__(self, csv_file, root_dir, mask_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.sessions_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return np.count_nonzero(self.sessions_frame.iloc[:,1:].values)

    def __getitem__(self, idx):
        # Flat indexing of tissues by acqusition session, skipping missing tissue samples not acquired in some sessions
        istissue = self.sessions_frame.iloc[:,1:].values
#        istissue = self.sessions_frame.iloc[:,[5,7]].values # 2-class problem of muscle vs fascia
        sub_row, sub_col = np.nonzero(istissue)
#        flat_ind = np.ravel_multi_index(np.nonzero(istissue), np.shape(istissue))
#        this_row, this_col = np.unravel_index(flat_ind[idx])
        this_row = sub_row[idx]
        this_col = sub_col[idx]
        this_session = self.sessions_frame.iloc[this_row, 0]
        this_tissue = self.sessions_frame.columns[this_col + 1]
        
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
        
        # Crop image to segmentation for visualization to confirm loading correct mask
        # Load mask of segmentation to limit cropping center for data augmentation transforms
        segmentation_folder = os.path.join(self.mask_dir, str(this_session))
        segmentation_filename = [f for f in os.listdir(segmentation_folder) if f.lower().endswith((this_tissue + 'Mask.png').lower())]
#        print(segmentation_folder)
#        print(this_tissue)
        assert len(segmentation_filename)>0, "Number of segmentation files is: " + str(len(segmentation_filename)) + " in folder " + str(segmentation_folder) + " for tissue " + str(this_tissue)
            
        segmentation = mpimg.imread(os.path.join(segmentation_folder, str(segmentation_filename[0])))
        segmentation = segmentation[:,:,0] # 2-dim mask of 0 or 1
        image, segmentation = get_segment_crop(image, mask=segmentation)   # crop image to bounding box around segmentation    
             
        # Convert tissue string names to categorical integers
        this_tissue = tissueclass_str2int(this_tissue)
        
        sample = {'image': image, 'tissue': this_tissue}
        
        if self.transform:
            
            # TODO: add crop to mask
            sample, segmentation = self.transform(sample, segmentation) # with functional transforms later
#            sample = self.transform(sample) # with functional transforms later

        return sample

    
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
        
def torchresize_preservedtype(image, output_size):
    """Rescale the image and preserve original image datatype and pixel range.

    Args:
        output_size (tuple): Desired output size. Output is
            matched to output_size. 
    """
    (new_h, new_w) = output_size
    if image.dtype=='float32' or image.dtype=='float64':
        img = transform.resize(image, (new_h, new_w), clip=False) # don't clip so negative values can be recovered after de-normalization
    else:
        info = np.iinfo(image.dtype) # Get the information of the incoming image type
        img = transform.resize(image, (new_h, new_w), clip=False) # converts image to dtype float64 in range [0, 1.0]
        img = info.max * img # Now scale by maximum of original dtype
        img = img.astype(info.dtype)
        
    return img
    

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

#        img = transform.resize(image, (new_h, new_w))
        img = torchresize_preservedtype(image, (new_h, new_w))

        return {'image': img, 'tissue': tissue}
    
    
class RandomResize(object):
    """Rescale the image in a sample to a random size. Random size (default: of 0.08 to 1.0) of the original size.
        Preserved aspect ratio.

    Args:
        scale (tuple): range of size of the origin size cropped. Default (0.08, 1.0) for Inception
    """

    def __init__(self, scale=(0.08, 1.0)):
        assert isinstance(scale, (int, tuple))
        self.scale = scale

    def __call__(self, sample):
        image, tissue = sample['image'], sample['tissue']

        h, w = image.shape[:2]
        
        random_scale = self.scale[0] + np.random.rand()*(self.scale[1] - self.scale[0])
        new_h, new_w = h * random_scale, w * random_scale

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'tissue': tissue}


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
        image, tissue = sample['image'], sample['tissue']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'tissue': tissue}
    

class RandomResizeSegmentation(object):
    """Rescale the image in a sample to a random size. Random size (default: of 0.08 to 1.0) of the original size.
        Preserved aspect ratio.

    Args:
        scale (tuple): range of size of the origin size cropped. Default (0.08, 1.0) for Inception
    """

    def __init__(self, scale=(0.08, 1.0)):
        assert isinstance(scale, (int, tuple))
        self.scale = scale

    def __call__(self, sample, segmentation):
        image, tissue = sample['image'], sample['tissue']

        h, w = image.shape[:2]
        
        random_scale = self.scale[0] + np.random.rand()*(self.scale[1] - self.scale[0])
        new_h, new_w = h * random_scale, w * random_scale

        new_h, new_w = int(new_h), int(new_w)

#        img = transform.resize(image, (new_h, new_w))
#        segmentation = transform.resize(segmentation, (new_h, new_w))
        
        img = torchresize_preservedtype(image, (new_h, new_w))
        segmentation = torchresize_preservedtype(segmentation, (new_h, new_w))

        return {'image': img, 'tissue': tissue}, segmentation

    
class RandomCropInSegmentation(object):
    """Crop randomly the image in a sample, within the segmentation mask ROI.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
            
        segmentation (2D-array): binarized mask of segmentation (0 or 1)
    """
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample, segmentation):
        image, tissue = sample['image'], sample['tissue']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        counter = 0
        insegmentation = False
        while not insegmentation:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            
            mid_row = top + new_h//2
            mid_col = left + new_w//2
            
            insegmentation = segmentation[mid_row, mid_col] > 0
            
            # Throw error if in infinite loop
            counter += 1
            assert counter < 10000, "Infinite loop in RandomCropInSegmentation transform: no crop of size " + str(self.output_size) + " found in image of size " + str(image.shape[:2]) + " after " + str(counter) + " random crop guesses in segmentation."

#        print(counter)
        image = image[top: top + new_h,
                      left: left + new_w]
        
        segmentation = segmentation[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'tissue': tissue}, segmentation


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, *kwargs):
        image, tissue = sample['image'], sample['tissue']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        if len(kwargs) == 0:
            return {'image': torch.from_numpy(image), 'tissue': tissue}
#                'landmarks': torch.from_numpy(landmarks)}
        else:
            segmentation = kwargs
            return {'image': torch.from_numpy(image), 'tissue': tissue}, segmentation        
    
class ImageStandardizePerDataset(object):
    """Normalize each Torch tensor image using pre-computed mean and standard deviation of all pixel values per channel across training dataset 6-5-19.

    Args:
        means (numpy rank2 array): mean pixel value per channel
        stds (numpy rank2 array): standard deviation of pixel values per channel

    """    
    
    def __init__(self, means=MEAN_CHANNEL_PIXELVALS, stds=STD_CHANNEL_PIXELVALS):
        self.channel_means = means
        self.channel_stds = stds

    def __call__(self, sample, segmentation):
        image, tissue = sample['image'], sample['tissue']
        
        if type(image) == torch.Tensor:
            im_tensortype = image.type()
            channel_means_torch = torch.from_numpy(self.channel_means).type(im_tensortype)
            channel_stds_torch = torch.from_numpy(self.channel_stds).type(im_tensortype)
            
            # swap color axis to ensure broadcast same dimensions in last dimension
            # numpy image: H x W x C
            # torch image: C X H X W
            image_whc = torch.transpose(image, 0, 2)
            normalized_image_whc = torch.sub(image_whc, channel_means_torch).div(channel_stds_torch)
            normalized_image = torch.transpose(normalized_image_whc, 0, 2) # C X H X W
        else:
            normalized_image = (image - self.channel_means) / self.channel_stds
            
        sample['image'] = normalized_image
        
        return sample, segmentation
    
class ImageStandardizePerImage(object):
    """Normalize each Torch tensor image using pre-computed mean and standard deviation of all pixel values per channel per image 6-5-19.

    Args:
        means (numpy rank2 array): mean pixel value per channel
        stds (numpy rank2 array): standard deviation of pixel values per channel

    """    
    def __call__(self, sample, segmentation):
        image, tissue = sample['image'], sample['tissue']
        
        if type(image) == torch.Tensor:
            inp = image.numpy().transpose((1, 2, 0))
            mean = np.array(np.mean(inp, axis=(0,1)))
            std = np.array(np.std(inp, axis=(0,1)))
            inp = (inp - mean)/std
            normalized_image = torch.from_numpy(inp.transpose((2, 0, 1)))
            
#            # Undo normalization
#            inp2 = normalized_image.numpy().transpose((1, 2, 0))
#            normalized_image = std * inp2 + mean
#            normalized_image = torch.from_numpy(inp2.transpose((2, 0, 1)))
#            print("Input image of type {}. Numpy image is of type {}. Normalized image is of type {}".format(image.dtype, inp.dtype, normalized_image.dtype))
            
             # Normalization method using all Tensor variables intermediate
#            # Ensure image is float tensor type to calculate standard deviation, uint8 for val data throws error
#            if not (image.dtype==torch.float or image.dtype==torch.float64):
#                image = image.type(torch.float64)
#            assert image.dtype==torch.float or image.dtype==torch.float64, "Image dtype is " + str(image.dtype) + " but expected torch.float32 or torch.float64 for torch.std"
#            
#            image_std = torch.zeros(image.size())
#            for i in np.arange(image.size()[0]):
#                image_std[i,:,:] = torch.std(image[i,:,:])
#            image_std = image_std.type(image.type()) # match data type of image for division
#            normalized_image = torch.sub(image, torch.mean(image, dim=(1,2), keepdim=True)).div(image_std) # C X H X W
##            normalized_image = torch.sub(image, torch.mean(image, dim=(1,2), keepdim=True)).div(torch.std(image, dim=(1,2), keepdim=True)) # C X H X W
        else:
            mean = np.array(np.mean(image, axis=(0,1)))
            std = np.array(np.std(image, axis=(0,1)))
            normalized_image = (image - mean) / std
            
#            # Undo normalization
#            normalized_image = std * normalized_image + mean
#            normalized_image = normalized_image.astype(image.dtype)
            print("Input image of type {}. Normalized image is of type {}".format(image.dtype, normalized_image.dtype))
            
        sample['image'] = normalized_image
        
        return sample, segmentation
    
class ImageUnstandardizePerDataset(object):
    """Un-Normalize each Torch tensor image using pre-computed mean and standard deviation of all pixel values per channel across training dataset 6-5-19.

    Args:
        means (numpy rank2 array): mean pixel value per channel
        stds (numpy rank2 array): standard deviation of pixel values per channel

    """    
    
    def __init__(self, means=MEAN_CHANNEL_PIXELVALS, stds=STD_CHANNEL_PIXELVALS):
        self.channel_means = means
        self.channel_stds = stds

    def __call__(self, sample, segmentation):
        image, tissue = sample['image'], sample['tissue']
        
        if type(image) == torch.Tensor:
            im_tensortype = image.type()
            channel_means_torch = torch.from_numpy(self.channel_means).type(im_tensortype)
            channel_stds_torch = torch.from_numpy(self.channel_stds).type(im_tensortype)
            
            # Match channel means and std to number of channels in image
            C = image.size()[0]
            channel_means_torch = channel_means_torch[0:C]
            channel_stds_torch = channel_stds_torch[0:C]
            
            # swap color axis to ensure broadcast same dimensions in last dimension
            # numpy image: H x W x C
            # torch image: C X H X W
            image_whc = torch.transpose(image, 0, 2) # C X H X W  -> H x W x C
            normalized_image_whc = torch.mul(image_whc, channel_stds_torch). add(channel_means_torch)
            normalized_image = torch.transpose(normalized_image_whc, 0, 2) # C X H X W
        else:
            normalized_image = (image * self.channel_stds) + self.channel_means
            
        sample['image'] = normalized_image
        
        return sample, segmentation
    
class ImageUnstandardizePerImage(object):
    """De-Normalize each Torch tensor image using pre-computed mean and standard deviation of all pixel values per channel per image 6-5-19.

    Args:
        means (numpy rank2 array): mean pixel value per channel
        stds (numpy rank2 array): standard deviation of pixel values per channel

    """    
    def __call__(self, sample, segmentation):
        image, tissue = sample['image'], sample['tissue']
        
        if type(image) == torch.Tensor:
            inp = image.numpy().transpose((1, 2, 0))
            mean = np.array(np.mean(inp, axis=(0,1)))
            std = np.array(np.std(inp, axis=(0,1)))
            inp = (inp * std) + mean
            normalized_image = torch.from_numpy(inp.transpose((2, 0, 1)))
            
#            # Undo normalization
#            inp2 = normalized_image.numpy().transpose((1, 2, 0))
#            normalized_image = std * inp2 + mean
#            normalized_image = torch.from_numpy(inp2.transpose((2, 0, 1)))
#            print("Input image of type {}. Numpy image is of type {}. Normalized image is of type {}".format(image.dtype, inp.dtype, normalized_image.dtype))
            
             # Normalization method using all Tensor variables intermediate
#            # Ensure image is float tensor type to calculate standard deviation, uint8 for val data throws error
#            if not (image.dtype==torch.float or image.dtype==torch.float64):
#                image = image.type(torch.float64)
#            assert image.dtype==torch.float or image.dtype==torch.float64, "Image dtype is " + str(image.dtype) + " but expected torch.float32 or torch.float64 for torch.std"
#            
#            image_std = torch.zeros(image.size())
#            for i in np.arange(image.size()[0]):
#                image_std[i,:,:] = torch.std(image[i,:,:])
#            image_std = image_std.type(image.type()) # match data type of image for division
#            normalized_image = torch.sub(image, torch.mean(image, dim=(1,2), keepdim=True)).div(image_std) # C X H X W
##            normalized_image = torch.sub(image, torch.mean(image, dim=(1,2), keepdim=True)).div(torch.std(image, dim=(1,2), keepdim=True)) # C X H X W
        else:
            mean = np.array(np.mean(image, axis=(0,1)))
            std = np.array(np.std(image, axis=(0,1)))
            normalized_image = (image * std) + mean
            
#            # Undo normalization
#            normalized_image = std * normalized_image + mean
#            normalized_image = normalized_image.astype(image.dtype)
            print("Input image of type {}. Normalized image is of type {}".format(image.dtype, normalized_image.dtype))
            
        sample['image'] = normalized_image
        
        return sample, segmentation
    
class ComposedTransform(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample, segmentation):
        for t in self.transforms:
            sample, segmentation = t(sample, segmentation)
        return sample, segmentation

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    
class getMeanStd(object):
    """Obtain mean and standard deviation of all pixel values per channel for pixel standardization per-dataset."""
    def __call__(self, sample, segmentation):
        image, tissue = sample['image'], sample['tissue']
        num_channels = np.shape(image)[2]
        assert num_channels == num_lights*3, "Number of image channels is " + str(num_channels) + " rather than expected number " + str(num_lights*3) + " for " + str(num_lights) + " illumination lights."
        
#        im_means = np.zeros((num_channels, 1))
#        im_stds = np.zeros((num_channels, 1))
        image_inmask = image[segmentation==True] # ndarray of shape (#mask pixels, num_channels)
        im_means = np.mean(image_inmask, axis=0)
        im_stds = np.std(image_inmask, axis=0)
        
        assert np.size(im_means)==num_channels, "Number of image means is " + str(np.size(im_means)) + " rather than expected number " + str(num_channels)
        assert np.size(im_stds)==num_channels, "Number of image means is " + str(np.size(im_stds)) + " rather than expected number " + str(num_channels)
        
        # Reshape arrays into rank2 numpy arrays
        im_means = np.reshape(im_means, (num_channels, 1))
        im_stds = np.reshape(im_stds, (num_channels, 1))
        
        sample_stats = {'means': im_means, 'stds': im_stds} 
        
        return sample_stats, segmentation
#        return im_means, im_stds    
    
def transform_denormalizeperdataset(image):
    """Stand alone transform to Un-Normalize image per dataset.
    
    Args:
        image: tensor (RGB or 21 channel) or numpy (21 channel) image to denormalize
    """ 
    denormalize_transform =  ImageUnstandardizePerDataset()
    sample = {'image': image, 'tissue': None}

    return denormalize_transform(sample, [])[0]['image']
    
           
#%% However, we are losing a lot of features by using a simple for loop to iterate over the data. In particular, we are missing out on:
#
#Batching the data
#Shuffling the data
#Load the data in parallel using multiprocessing workers.
#torch.utils.data.DataLoader is an iterator which provides all these features. Parameters used below should be clear. One parameter of interest is collate_fn. You can specify how exactly the samples need to be batched using collate_fn. However, default collate should work fine for most use cases.

# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with tissues for a batch of samples."""
    images_batch, tissues_batch = \
            sample_batched['image'], sample_batched['tissue']
    batch_size = len(images_batch)

    grid = utils.make_grid(images_batch[:,:3,:,:]) # 4D mini-batch Tensor of shape (B x C x H x W) ; use RGB channels only
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
##        plt.title(tissues_batch)
#        plt.title(tissueclass_int2str(tissues_batch.numpy())[i])
        plt.title(tissues_batch.numpy())
        plt.title(itemgetter(*tissues_batch)(myclasses))

# Helper function to plot histograms of image pixel data per RGB channels
def show_histogram_RGBchannels(image):
    image_r = image.data[0,:,:].flatten()
    image_g = image.data[1,:,:].flatten()
    image_b = image.data[2,:,:].flatten()
    
    plt.figure()
    rgbcolors = ['r', 'g', 'b']
#        plt.hist([image_r, image_g, image_b], bins='auto', color=rgbcolors) # histogram plot side-by-side
    plt.hist(image_r, bins='auto', alpha=0.5, label='R', color=rgbcolors[0])
    plt.hist(image_g, bins='auto', alpha=0.5, label='G', color=rgbcolors[1])
    plt.hist(image_b, bins='auto', alpha=0.5, label='B', color=rgbcolors[2])
    plt.legend(loc='upper right')
    plt.ylabel('Count')
    plt.xlabel('Pixel value')
    plt.show()
    

def main():
    
    #%% Let’s quickly check number of training images per tissue type.
#    sessions_frame = pd.read_csv(os.path.join(PATH_DATA, 'arritissue_sessions.csv'))

    train_sessions = [f for f in listdir(PATH_TRAIN) if isdir(join(PATH_TRAIN, f))] # Get only folders
    #val_sessions = [f for f in listdir(PATH_VAL) if isdir(join(PATH_VAL, f))]
    num_sessions = len(train_sessions)
#    session_tissues = np.zeros((num_sessions, 1))
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
        
    #%% Show sample image with helper function
    plt.figure()
    session_i = 0 # Date of acquisition
    tissue_j = 2 # tissue type index
    path_sessioni = os.path.join(PATH_TRAIN, train_sessions[session_i])
    this_folder = os.path.join(path_sessioni, acquisition_labels[tissue_j])
    this_tiffs = [f for f in os.listdir(this_folder) if f.endswith('.tif')]
    trial_image = mpimg.imread(os.path.join(this_folder, this_tiffs[0])) # arriwhite image
    show_tissue(trial_image, acquisition_labels[tissue_j])
    plt.show()
    
    #%% Show sample image cropped to segmentation
    # Crop image to segmentation for visualization to confirm loading correct mask
    # Load mask of segmentation to limit cropping center for data augmentation transforms
    segmentation_folder = os.path.join(PATH_MASK, str(train_sessions[session_i]))
    segmentation_filename = [f for f in os.listdir(segmentation_folder) if f.lower().endswith((acquisition_labels[tissue_j] + 'Mask.png').lower())]
    segmentation = mpimg.imread(os.path.join(segmentation_folder, str(segmentation_filename[0])))
    segmentation = segmentation[:,:,0] # 2-dim mask of 0 or 1
    trial_image_cropped, segmentation = get_segment_crop(trial_image, mask=segmentation)   
    show_tissue(trial_image_cropped, acquisition_labels[tissue_j])
    plt.show()     
    
    # Show segmentation
    plt.figure()
    show_tissue(np.repeat(segmentation[:, :, np.newaxis], 3, axis=2), acquisition_labels[tissue_j])
    plt.show()     
    
    #%% instantiate this class and iterate through the data samples. We will print the sizes of first 4 samples and show their landmarks.
    tissue_dataset = ArriTissueDataset(csv_file=PATH_CSV, root_dir=PATH_TRAIN, mask_dir=PATH_MASK)
    
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
    
    #%% Compose transforms
    #Now, we apply the transforms on an sample.
    #
    #Let’s say we want to rescale the shorter side of the image to 256 and then randomly crop a square of size 224 from it. 
    # i.e, we want to compose Rescale and RandomCrop transforms. torchvision.transforms.Compose is a simple callable class which allows us to do this.
    scale = Rescale((1080, 1920))
#    randomrescale = RandomResize()
#    crop = RandomCrop(32)
#    composed = transforms.Compose([randomrescale, crop])
    randomrescale = RandomResizeSegmentation()
    crop = RandomCropInSegmentation(32)
    normalize = ImageStandardizePerDataset()
    denormalize = ImageUnstandardizePerDataset()
#    normalize = ImageStandardizePerImage()
#    denormalize = ImageUnstandardizePerImage()
    composed = ComposedTransform([randomrescale, crop, normalize, denormalize])
    
#    # PyTorch Vision Transforms only work on PILLOW (PIL) images, not good for multichannel (>3) images! 
#    crop2 = transforms.RandomCrop(32)
#    resizecrop = transforms.RandomResizedCrop(32) # torchvision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)
#    rotate = transforms.RandomRotation(30)
#    composed2 = transforms.Compose([rotate, resizecrop])
    
    # Apply each of the above transforms on sample.
    fig = plt.figure()
    sample = tissue_dataset[2]
#    for i, tsfrm in enumerate([scale, crop, composed]):
    list_tsfrms = [randomrescale, crop, normalize, composed]
    for i, tsfrm in enumerate(list_tsfrms):
#        transformed_sample = tsfrm(sample)
        print(np.shape(sample['image']), np.shape(segmentation))
#        print('Max per channel: {}'.format(np.amax(sample['image'], axis=(0,1))))
#        print('Min per channel: {}'.format(np.amin(sample['image'], axis=(0,1))))
        output = tsfrm(sample, segmentation)
        if np.size(output) == 2:
            transformed_sample = output[0]
            transformed_segmentation = output[1]
        elif np.size(output) == 1:
            transformed_sample = output
            
#    for i, tsfrm in enumerate([crop2, rotate, resizecrop, composed2]):
#        tsfrm_image = tsfrm(sample['image'])
#        transformed_sample =  {'image': tsfrm_image, 'tissue': sample['tissue']}
    
        ax = plt.subplot(1, len(list_tsfrms), i + 1)
#        plt.tight_layout()
        show_tissue(**transformed_sample)
        ax.set_title(type(tsfrm).__name__)
    
    plt.show()
    
    #%% Iterating through the dataset
    #Let’s put this all together to create a dataset with composed transforms. To summarize, every time this dataset is sampled:
    #
    #An image is read from the file on the fly
    #Transforms are applied on the read image
    #Since one of the transforms is random, data is augmentated on sampling
    #We can iterate over the created dataset with a for i in range loop as before.
    
    transformed_dataset = ArriTissueDataset(csv_file=PATH_CSV, root_dir=PATH_TRAIN, mask_dir=PATH_MASK,
#                                               transform=transforms.Compose([
                                                transform=ComposedTransform([
#                                                   Rescale((1080, 1920)),
#                                                   RandomResize(),
#                                                   RandomCrop(32),
                                                   RandomResizeSegmentation(),
                                                   RandomCropInSegmentation(32),
                                                   ToTensor(),
                                                   normalize,
#                                                   denormalize
#                                                   ImageStandardizePerImage()
                                               ]))
    
    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
    
        print(i, sample['image'].size(), myclasses[sample['tissue']])
        
        image = sample['image']
#        print('Max val {}, {}, {} and min val {}, {}, {}'.format(
#            torch.max(image.data[0, :, :]), torch.max(image.data[1, :, :]), torch.max(image.data[2, :, :]),
#            torch.min(image.data[0, :, :]), torch.min(image.data[1, :, :]), torch.min(image.data[2, :, :])))
        
        # Plot histogram of pixel values after normalization
        show_histogram_RGBchannels(image)
    
        if i == 13:
            break
        
    #%% Use torch.utils.data.DataLoader is an iterator which provides all these features. Parameters used below should be clear. One parameter of interest is collate_fn. You can specify how exactly the samples need to be batched using collate_fn. However, default collate should work fine for most use cases.
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)
    
    time_start = time.time()
    print('Dataloader getting first batch... ')
    for i_batch, sample_batched in enumerate(dataloader):
        print('Time elapsed: {} seconds'.format(time.time()-time_start))
        
        print(i_batch, sample_batched['image'].size(),
              itemgetter(*sample_batched['tissue'])(myclasses))
    
        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break
        
        
#    #%% 6-4-19: Compute mean and std of 21 channels across training dataset to standardize images for transfer learning
#    dataset_stats = ArriTissueDataset(csv_file=PATH_CSV, root_dir=PATH_TRAIN, mask_dir=PATH_MASK, transform=getMeanStd())
#    running_means = np.zeros((21, 1))
#    running_stds = np.zeros((21, 1))
#    for i in range(len(dataset_stats)):
#        sample = dataset_stats[i]
#    
#        print(i, sample['means'], sample['stds'])
#    
#        im_mean = sample['means']
#        im_std = sample['stds']
#        
#        running_means += im_mean
#        running_stds += im_std
#
#    running_means /= len(dataset_stats)        
#    running_stds /= len(dataset_stats)
#
#    print('Mean pixel value per channel: {}'.format(running_means))
#    print('Std of pixel values per channel: {}'.format(running_stds))

if __name__=='__main__':
    main()
