import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from Gaussian import gaussian
from saltAndPepper import saltNpepper


def median(image,kernelDim):
    dim = (kernelDim-1)//2
    height,width = image.shape
    for i in range(0,height):
        for j in range(0,width):
            a = np.array([])
            iprim = i
            jprim = j
            if iprim - dim < 0: # ako ce krajnji deo kernel matrice izaci iz range-a, treba da se shiftuje ka unutra
                iprim = dim

            if jprim - dim < 0:
                jprim = dim

            if iprim + dim > height - 1:
                iprim = height -1 - dim

            if jprim + dim > width - 1:
                jprim = width -1 - dim
            for k in range(-dim,dim+1):
                for l in range(-dim,dim+1): 
                    a = np.append(a,image[iprim + k][jprim + l])
            
            a = np.sort(a,axis = None)
           
            image[i][j] = int(a[len(a)//2])
            
    return image

'''img = cv2.imread('kupusii.png',0)
noisy = saltNpepper(img,5)
cv2.imshow('NoisyImage',noisy)

filtered = median(noisy,3)
#print(filtered.shape)
cv2.imshow('Filtered',filtered)
k = cv2.waitKey(0) & 0xFF
if k == ord('q'):        
    cv2.destroyAllWindows()'''