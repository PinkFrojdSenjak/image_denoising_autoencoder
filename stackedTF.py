from keras.models import Sequential
from keras.layers import Input , Dense , Dropout
from keras.models import Model
import numpy as np
import tensorflow as tf
from tensorflow import keras
from ucitavanje_baze import ucitaj_bazu
import os
import cv2
from Gaussian import gaussian
from saltAndPepper import saltNpepper
from tensorflow.keras import backend as K
import math



def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 *  K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303

def flatten(a):
    b = np.array([])
    for i in a:
        for j in i:
            b = np.append(b, j)
    b = np.reshape(b, (len(b),1))
    return b/255


baza_images = []
baza_labels = []


for folder in os.listdir("slike_baza"):
    for filename in os.listdir(os.path.join("slike_baza", folder)):
        t = cv2.imread(os.path.join(os.path.join("slike_baza", folder), filename),0)

        x = np.copy(t)
        noisy = saltNpepper(x, 3)

        for i in range(0,len(t),32):
            for j in range(0,len(t[0]),32):
                a = t[min(i, len(t)-32):min(i+32, len(t)), min(j, len(t[0])-32):min(j+32,len(t[0]))] 
                b = noisy[min(i, len(t)-32):min(i+32, len(t)), min(j, len(t[0])-32):min(j+32,len(t[0]))]
                baza_images.append(b.flatten()/ 255)
                baza_labels.append(a.flatten()/255)
            

print("ucitao")

baza_images = np.array(baza_images)
baza_labels = np.array(baza_labels)

train_images = baza_images[:len(baza_images)-1560]
test_images = baza_images[len(baza_images)-1560:]   


train_labels = baza_labels[:len(baza_labels)-1560]
test_labels = baza_labels[len(baza_labels)-1560:]


print(train_images.shape)

inputImg = Input( shape = (1024, ) )
encoded =  Dense (units = 256 , activation = 'relu') (inputImg)
decoded =  Dense (units = 1024 , activation = 'sigmoid') (encoded)
decoded = Dropout(0.15, noise_shape=None, seed=None) (decoded)

autoEncoder = Model(inputImg , decoded)

encoder = Model(inputImg , encoded)

autoEncoder.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = [PSNR])

autoEncoder.fit(train_images, train_labels, epochs=5, batch_size=40, shuffle = True, validation_data = (test_images, test_labels))


encoded_imgs = encoder.predict(train_images)

img = autoEncoder.predict(train_images) * 255 

def spoji_sliku(x):

    br = 64*x

    slika = np.reshape(img[br], (32,32))

    s = np.array(slika, dtype = np.uint8)

    b = np.copy(s)

    br+=1

    for j in range(1, 8):
        slika = np.reshape(img[br], (32,32))

        s = np.array(slika, dtype = np.uint8)

        b = np.concatenate((b, s), axis = 1)

        br+=1

    a = np.copy(b)


    for i in range(1,8):
        slika = np.reshape(img[br], (32,32))
        s = np.array(slika, dtype = np.uint8)
        b = np.copy(s)
        
        br+=1

        for j in range(1, 8):
            slika = np.reshape(img[br], (32,32))
            s = np.array(slika, dtype = np.uint8)

            b = np.concatenate((b, s), axis = 1)

            br+=1

        a = np.concatenate((a, b), axis = 0)

    return(a)


cv2.imshow('slika', spoji_sliku(1))
cv2.waitKey(0)

cv2.imshow('slika', spoji_sliku(2))
cv2.waitKey(0)

encoded_imgs = np.concatenate((encoded_imgs, encoder.predict(train_labels) ) , axis = 0)



print("ko adrijana lima hodom provocira")
