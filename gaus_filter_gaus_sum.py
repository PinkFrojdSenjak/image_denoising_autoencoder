import cv2
import numpy as np
from gaussianFilter import gausov_filter
from Gaussian import gaussian
from PSNR import psnr
from ucitavanje_baze import ucitaj_bazu

baza = ucitaj_bazu()

rezPre = 0
rez = 0
br = 0

x = 0

for img in baza:
    x += 1
    if x%5 > 0 :
        continue

    img2 = np.copy(img)

    noisy = gaussian(img, 3000)
    
    filtered = gausov_filter(noisy, 3)

    rez += psnr(img, filtered)
    rezPre += psnr(img2, noisy)
    br += 1

rez /= br
rezPre /= br

print(rez)
print(rezPre)