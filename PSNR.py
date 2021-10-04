import cv2
import math
import numpy as np

def psnr(img1, img2):

    n,m = img1.shape

    mse = 0.0
    img1cpy = np.array(img1,dtype = np.float)
    img2cpy = np.array(img2,dtype = np.float)    
    for i in range(n):
        for j in range(m):
            mse += (img1cpy[i][j]-img2cpy[i][j])**2 / (m*n)
            
    if mse==0:
        return 2
    return 10 * math.log(255*255/mse, 10)



