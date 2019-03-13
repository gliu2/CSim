# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:32:46 2019

@author: George Liu

Interactively segment all images in a folder. Loads images in random order.
-hold left button while moving mouse to draw
-paintbrush locks to border when mouse leaves window
-press backspace to reset current image
-press enter or escape key to save segmentation and open next file in folder

Input: path to directory containing image files (of # num_files)
Output: Saves following variables in pickle file 'GLsegment_objs.pkl':
            cache_points - list of lists containing (x,y) coordinates of images' segmentations
            cache_time - time (seconds) for segmentation per image, numpy array of shape (# num_files, 1)
            onlyfiles - list of image files names
            shuffled_order - list of image order

Folder path is hardcoded. Please ensure only image files are in folder.

Last edit: 3/13/2019 

Dependencies: none
"""

import cv2
import numpy as np 
import os
import random
import time
import pickle

drawing=False # true if mouse is pressed
mode=True # if True, draw as paintbrush. Press 'm' to toggle
BresenhamConnectivity=4
LineThickness=2 # reviewed with jared

points = []

# mouse callback function
def paintbrush_draw(event, x, y,flags,param):
    global former_x, former_y, drawing, mode

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        former_x, former_y = x, y
        if mode==True:
            cv2.line(im,(former_x, former_y),(x, y),(0,0,255), thickness=LineThickness, lineType=BresenhamConnectivity)
            points.append((x,y))

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                # Set mouse position to image border if mouse leaves window
                if x<0:
                    x=0
                elif x>(im.shape[1]-1):
                    x=(im.shape[1]-1)
                    
                if y<0:
                    y=0
                elif y > (im.shape[0]-1):
                    y = (im.shape[0]-1)

                # Draw line                
                cv2.line(im,(former_x, former_y),(x, y),(0,0,255), thickness=LineThickness, lineType=BresenhamConnectivity)
                former_x = x
                former_y = y
                points.append((x,y))
                
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
            
    return x, y            

#%%  Path to folder containing images to segment interactively
mypath = 'C:\\Users\\CTLab\\Documents\\George\\ELH_images'
#mypath = 'C:\\Users\\CTLab\\Documents\\George\\test_images'
onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
num_files = len(onlyfiles)
cache_points = []
cache_time = np.zeros((num_files, 1))

# Shuffle order of loading images
shuffled_order = [[i] for i in range(num_files)]
random.shuffle(shuffled_order)

#%% Run interactive drawing
for index, _ in enumerate(onlyfiles):
    f = onlyfiles[shuffled_order[index][0]] # load image in random order
    im = cv2.imread(os.path.join(mypath, f))
    im_copy = im.copy()
    cv2.namedWindow("Segment OpenCV")
    cv2.setMouseCallback('Segment OpenCV', paintbrush_draw)
    
    # start time for segmentation
    start = time.time() 
    
    while(1):        
        cv2.imshow('Segment OpenCV', im)
        
        # Obtain interactive keyboard input
        k=cv2.waitKey(1)&0xFF  
        
        # Reset image if hit backspace
#        if k == ord('BS'): # backspace
        if k == 8: # backspace
            # reset points
            points = []
            
            # reset image
            im = im_copy.copy()
    
        # Finish if hit ESC key or Enter key
        if (k==27) or (k==13): 
            break
    
    # record time for segmentation (seconds)
    end = time.time()
    runtime = end - start
    cache_time[index] = runtime
    
    # cache segmented coordinates
    cache_points.append(points)
    
    # reset list of segmented points
    points = []
    
cv2.destroyAllWindows()

# Saving the objects:
with open('GLsegment_objs.pkl', 'wb') as ff:  # Python 3: open(..., 'wb')
    pickle.dump([onlyfiles, shuffled_order, cache_points, cache_time], ff)
    
    
## Getting back the objects:
#with open('GLsegment_objs.pkl', 'rb') as ff:  # Python 3: open(..., 'rb')
#    onlyfiles, shuffled_order, cache_points, cache_time = pickle.load(ff)