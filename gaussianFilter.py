import cv2
import numpy as np
from Gaussian import gaussian
from saltAndPepper import saltNpepper
from PSNR import psnr
#picture = cv2.imread('kupusii.png', 0)

def mk_okvir(picture):
    img = np.zeros(( len(picture)+4, len(picture[0])+4))

    for i in range(len(img)):
        for j in range(len(img[0])):
            if i<=1:
                if j<=1:
                    img[i][j] = picture[0][0]
                    continue
                if j>=len(img[0])-2:
                    img[i][j] = picture[0][len(picture[0])-1]
                    continue
                img[i][j] = picture[0][j-2]
                continue
            if i>=len(img)-2:
                 
                if j<=1:
                    img[i][j] = picture[len(picture)-1][0]
                    continue
                if j>=len(img[0])-2:
                    img[i][j] = picture[len(picture)-1][len(picture[0])-1]
                    continue
                img[i][j] = picture[len(picture)-1][j-2]
                continue
            if j<=1:
                img[i][j] = picture[i-2][0]
                continue
            if j>=len(img[0])-2:
                img[i][j] = picture[i-2][len(picture[0])-1]
                continue
            img[i][j] = picture[i-2][j-2]
    return img


def gausov_filter(picture,size = 5):

    pic = np.zeros(( len(picture), len(picture[0])))

    picture = mk_okvir(picture)

    sigma = 1
    #size = 5

    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    gausov_kernel =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

    #pic = cv2.GaussianBlur(picture,(5,5),0)



    for i in range(2, len(picture)-2):
        for j in range(2, len(picture[0])-2):
            for x in range(0, len(gausov_kernel)):
                for y in range(0, len(gausov_kernel[0])):
                    pic[i-2][j-2]=pic[i-2][j-2]+picture[i+x-2][j+y-2]*gausov_kernel[x][y]
            pic[i-2][j-2]=int(pic[i-2][j-2]/gausov_kernel.sum())



    return pic

'''img = cv2.imread('kupusii.png',0)
noisy = gaussian(img,1000)
noisy = saltNpepper(img,5)
cv2.imshow('NoisyImage',noisy)
#cv2.waitKey(0)


filtered = gausov_filter(noisy)


s = np.array(filtered, dtype=np.uint8)
cv2.imshow('Gaus',s)
cv2.waitKey(0)

print(psnr(noisy, filtered))'''
