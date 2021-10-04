import cv2
import numpy as np
from medianFilter import median
from saltAndPepper import saltNpepper
from PSNR import psnr
from ucitavanje_baze import ucitaj_bazu

baza = ucitaj_bazu()

rez = 0
br = 0
x = 0
rezPre = 0
for img in baza:
    x+=1
    if x%5>0:
        continue
    img2 = np.copy(img)
    noisy = saltNpepper(img2,5)
    noisy2 = np.copy(noisy)
    rezPre += psnr(img,noisy)
    filtered = median(noisy2, 3)
    rez += psnr(img, filtered)
    #print(psnr(img, filtered))
    br += 1

rezPre /= br
rez /= br

print(rez,rezPre)