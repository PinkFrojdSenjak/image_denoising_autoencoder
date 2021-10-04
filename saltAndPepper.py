import cv2
import numpy as np
import random

# ovo je kod sa funkcijom za salt n pepper sum,
# a dole pod komentarom je samo kod za zasumljivanje date slike i prikazivanje

def saltNpepper(image, procenatSuma):
    maxrand = 1000
    height,width = image.shape
    #noisy_image = np.zeros(image.shape,np.uint8)
    noisy_image = np.copy(image, np.uint8)

    for i in range(height):
        for j in range(width):
            r = random.randrange(1,maxrand+1)
            if r>maxrand*(1-procenatSuma/100):
                randColor = random.randrange(0,2)
                randColor *= 255
                noisy_image[i][j] = randColor
    
    return noisy_image
'''
procenat = 3
img = cv2.imread('kupusii.png',0)
noisyImg = saltNpepper(img,procenat)

cv2.imshow('NoisyImage',noisyImg)

k = cv2.waitKey(0) & 0xFF
if k == ord('q'):        
    cv2.destroyAllWindows()
'''
