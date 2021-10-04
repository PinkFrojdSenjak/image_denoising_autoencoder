import autoencoder
from ucitavanje_baze import ucitaj_bazu
import os
import cv2
import numpy as np
from Gaussian import gaussian

def flatten(a):
    b = np.array([])
    for i in a:
        for j in i:
            b = np.append(b, j)
    b = np.reshape(b, (len(b),1))
    return b/255

baza = []

for folder in os.listdir("slike_baza"):
    for filename in os.listdir(os.path.join("slike_baza", folder)):
        t = cv2.imread(os.path.join(os.path.join("slike_baza", folder), filename),0)
        for i in range(0,len(t),32):
            for j in range(0,len(t[0]),32):
            
                a = t[min(i, len(t)-32):min(i+32, len(t)), min(j, len(t[0])-32):min(j+32,len(t[0]))]
                b = gaussian(a, 1000)
                baza.append((flatten(a),flatten(b)))


print(len(baza))

training_data = baza[:len(baza)-1560]
test_data = baza[len(baza)-1560:]    



model = autoencoder.Net([1024, 512, 1024])
model.fit_model(training_data,test_data,10,10)





