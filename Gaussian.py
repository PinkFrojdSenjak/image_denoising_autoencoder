import cv2
import numpy as np
import random



def gaussian(image,var):
    height,width = image.shape

    mean = 0
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (height, width)) #  np.zeros((224, 224), np.float32)

    noisy_image = np.zeros(image.shape, np.float32)

    if len(image.shape) == 2:
        noisy_image = image+ gaussian            

    outImg = np.clip(noisy_image,0,255)
    outImg = outImg.astype(np.uint8)
    return outImg




'''img = cv2.imread('kupusii.png',0)
var = 1000 # staviti nesto reda velicine 1000
outImage = gaussian(img,var)

cv2.imshow('NoisyImage',outImage)

cv2.imshow('Image',img)
k = cv2.waitKey(0) & 0xFF
if k == ord('q'):        
    cv2.destroyAllWindows()'''