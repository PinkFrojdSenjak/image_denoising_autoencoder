import numpy as np

a = np.array([ [ [1,2,3],[1,2,3],[1,2,3] ] ])

b = np.array([ [ [4,5] ] ,[ [4,5] ] ])

print(np.concatenate((a,b),axis = None) )

