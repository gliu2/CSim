# -*- coding: utf-8 -*-
"""
Created on Apr 3, 2016

@author: Bill BEGUERADJ
"""

import cv2
import numpy as np 

drawing=False # true if mouse is pressed
mode=True # if True, draw rectangle. Press 'm' to toggle to curve
BresenhamConnectivity=4
LineThickness=1

points = []
counter=0

# mouse callback function
def begueradj_draw(event, x, y,flags,param):
    global former_x, former_y, drawing, mode, im, counter

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        former_x, former_y = x, y
        if mode==True:
#            cv2.circle(im, (x,y), 1, (0,0,255),-1)
#            points.append((x,y))
#            pts = np.array(points, np.int32)
#            cv2.polylines(im,[pts],False,(0,0,255))
#        else:
            cv2.line(im,(former_x, former_y),(x, y),(0,0,255), thickness=LineThickness, lineType=BresenhamConnectivity)
            points.append((x,y))
            


    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.line(im,(former_x, former_y),(x, y),(0,0,255), thickness=LineThickness, lineType=BresenhamConnectivity)
                former_x = x
                former_y = y
                points.append((x,y))
                
#                print(x, y)
#                next_point = (x, y)
#                this_line = this_line.append(next_point)
                
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
#        if mode==False:
#            cv2.line(im,(former_x, former_y),(x, y),(0,0,255), thickness=LineThickness, lineType=BresenhamConnectivity)
#            former_x = x
#            former_y = y

    elif event==cv2.EVENT_RBUTTONDOWN:
        if drawing==False:
            del points[-1]
            im = im_original
            former_x, former_y = points[0]
            im = im_original.copy()
#            counter = counter + 1
#            newwindow_name = 'Segment' + str(counter)
#            cv2.namedWindow(newwindow_name)
#            cv2.imshow(newwindow_name,im)
            cv2.imshow('Bill BEGUERADJ OpenCV',im)
            for x, y in points:
                cv2.line(im,(former_x, former_y),(x, y),(0,0,255), thickness=LineThickness, lineType=BresenhamConnectivity)
                former_x, former_y = x, y
            
    return x, y            


im_original = cv2.imread("darwin.jpg")
im = im_original.copy() # copy for undo feature
cv2.namedWindow("Bill BEGUERADJ OpenCV")
cv2.setMouseCallback('Bill BEGUERADJ OpenCV',begueradj_draw)
while(1):
    cv2.imshow('Bill BEGUERADJ OpenCV',im)
    k=cv2.waitKey(1)&0xFF  
#    if k == ord('BS'): # backspace
#        mode = not mode
    if k == ord('m'):
        mode = not mode
    if (k==27) or (k==13): # ESC key or Enter key
        break
cv2.destroyAllWindows()