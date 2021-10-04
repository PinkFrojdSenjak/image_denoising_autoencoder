import cv2
import os


def ucitaj_bazu():
    baza = []

    for folder in os.listdir("slike_baza"):
        for filename in os.listdir(os.path.join("slike_baza", folder)):
                t = cv2.imread(os.path.join(os.path.join("slike_baza", folder), filename),0)
                baza.append(t)
    
    return baza

