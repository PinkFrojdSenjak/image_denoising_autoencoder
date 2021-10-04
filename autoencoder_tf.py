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
        for i in range(0,len(t),32):
            for j in range(0,len(t[0]),32):
                a = t[min(i, len(t)-32):min(i+32, len(t)), min(j, len(t[0])-32):min(j+32,len(t[0]))] 
                b = saltNpepper(a, 3)
                baza_images.append(a/ 255)
                baza_labels.append(b.flatten())
            

print("ucitao")

baza_images = np.array(baza_images)
baza_labels = np.array(baza_labels)

train_images = baza_images[:len(baza_images)-1560]
test_images = baza_images[len(baza_images)-1560:]   


train_labels = baza_labels[:len(baza_labels)-1560]
test_labels = baza_labels[len(baza_labels)-1560:]


train_images.reshape(len(train_images), 1024, 1)
train_labels.reshape(len(train_labels), 1024, 1)

print(train_images.shape)


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32)),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(1024, activation=tf.nn.relu)
])

model.compile(optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.001), 
              loss='mean_squared_error',
              metrics=[PSNR])


test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

model.fit(train_images, train_labels, epochs=20, batch_size=40)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

model.save_weights("model.h5")




#generate data
'''no = 100
data_x = np.linspace(0,1,no)
data_y = 2 * data_x + 2 + np.random.uniform(-0.5,0.5,no)
data_y = data_y.reshape(no,1)
data_x = data_x.reshape(no,1)'''

'''
# Make model using keras layers and train
x = tf.placeholder(dtype=tf.float32, shape=[None,1])
y = tf.placeholder(dtype=tf.float32, shape=[None,1])

h = tf.keras.layers.Dense(512, activation='sigmoid')(x)  
output = tf.keras.layers.Dense(1024, activation='sigmoid')(h)

loss = tf.losses.mean_squared_error(train_labels[0], output)
optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method="L-BFGS-B")

sess = K.get_session()
sess.run(tf.global_variables_initializer())

tf_dict = {x : train_images[0], y : train_labels[0]}
optimizer.minimize(sess, feed_dict = tf_dict, fetches=[loss], loss_callback=lambda x: print("Loss:", x))'''