import cv2
import numpy as np
from matplotlib import pyplot as plt

# Defining the camera projection matrices for the stereo cameras used
p_left = [[640.0,   0.0, 640.0, 2176.0], 
         [0.0, 480.0, 480.0,  552.0], 
         [0.0,   0.0,   1.0,    1.4]]
p_right = [[640.0,   0.0, 640.0, 2176.0], 
          [0.0, 480.0, 480.0,  792.0], 
          [0.0,   0.0,   1.0,    1.4]]

# Images taken from cameras
limg=cv2.imread("left.png",cv2.IMREAD_GRAYSCALE)
rimg=cv2.imread("right.png",cv2.IMREAD_GRAYSCALE)
obs=cv2.imread("bike.png",cv2.IMREAD_GRAYSCALE)

# Template Matching to detect the object in the images from the left and right cameras

threshold = 0.97
res = cv2.matchTemplate(limg,obs,cv2.TM_CCOEFF_NORMED)
loc = np.where(res >= threshold)
if(len(loc[0]) != 0 and len(loc[1]) != 0):
    for pt in zip(*loc[::-1]):
        cv2.rectangle(limg, pt, (pt[0] + obs.shape[1], pt[1] + obs.shape[0]), (0,255,255), 3)
        cv2.imshow("template_left",limg)
        xl=pt[0] + obs.shape[0]
        yl=pt[1] + obs.shape[1]

res = cv2.matchTemplate(rimg,obs,cv2.TM_CCORR_NORMED)
loc = np.where(res >= threshold)
if(len(loc[0]) != 0 and len(loc[1]) != 0):
    for pt in zip(*loc[::-1]):
        cv2.rectangle(rimg, pt, (pt[0] + obs.shape[1], pt[1] + obs.shape[0]), (0,255,255), 3)
        cv2.imshow("template_right",rimg)
        xr=pt[0] + obs.shape[0]   
        yr=pt[1] + obs.shape[1]

disparity = xr-xl      # The x difference between corresponding points of 2 images

f = -p_left[0][0]                                   # Focal Length
B = (p_right[1][3]-p_left[1][3])/p_left[1][1]       # Distance between the cameras

dist=f*B/disparity
print("Distance = ",dist)

cv2.waitKey(0)
cv2.destroyAllWindows()
