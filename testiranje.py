import numpy as np
import random
import tensorflow
from tensorflow import keras

b = np.array([[]])

a = np.array([[1,2,3],[4,5,6],[7,8,9]] )

print(b.shape)
print(a.shape)

a = np.concatenate((a,a),axis = 1)


print(a)


a = np.concatenate((a,a), axis = 1)

print(a)





