import cv2
import numpy as np
from medianFilter import median
from Gaussian import gaussian
from PSNR import psnr
from ucitavanje_baze import ucitaj_bazu

baza = ucitaj_bazu()

rez = 0
br = 0
rezPre = 0
x= 0 

for img in baza:
    x+=1
    if x%5>0:
        continue
    img2 = np.copy(img)
    noisy = gaussian(img2, 2000)
    noisy2 = np.copy(noisy)
    rezPre += psnr(noisy, img)
    filtered = median(noisy2,7)
    rez += psnr(img, filtered)
    br += 1

rez /= br
rezPre /= br
print(rezPre , rez)
