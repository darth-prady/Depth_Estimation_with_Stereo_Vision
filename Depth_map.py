import cv2
import numpy as np
from matplotlib import pyplot as plt

limg=cv2.imread("left.png",cv2.IMREAD_GRAYSCALE)
rimg=cv2.imread("right.png",cv2.IMREAD_GRAYSCALE)

# Plotting the Depth Map to visualize the distances of different objects

stereo = cv2.StereoBM_create(numDisparities=32, blockSize=13)
depthmap = stereo.compute(limg,rimg)

plt.title("Depth Map")
plt.imshow(depthmap,"gray")
plt.show()