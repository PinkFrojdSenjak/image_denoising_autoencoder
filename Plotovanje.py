import numpy as np
import matplotlib.pyplot as plt

'''
x = np.array([ 3 , 5 , 7 ])
y = np.array([ 27.8437 , 24.7169 , 22.9465 ])
plt.title("Median filter - Salt n Pepper - sum: 3%")
'''

'''
x = [3,5,7]
y = [ 27.7609,24.5559,22.6437 ]
plt.title("Median filter - Salt n Pepper - sum: 5%")
'''
'''
x = [3,5,7]
y = [27.2561 , 24.0314 , 21.9841 ]
plt.title("Median filter - Salt n Pepper - sum: 10%")
'''
'''
x = [3,5,7]
y = [24.5733 , 22.2387 , 20.4577]
plt.title("Median filter - Gaus Sum - jacina suma: 1000")
'''
'''
x = [3,5,7]
y = [ 23.3548 , 21.1869 , 19.2836 ]
plt.title("Median filter - Gaus Sum - jacina suma: 2000")
'''
'''
x = [3,5,7]
y = [22.5173 , 20.4789 , 18.6481]
plt.title("Median filter - Gaus Sum - jacina suma: 3000")
'''

'''
x = [3,5,7]
y = [22.2093 , 25.4237 , 20.0937]
plt.title("Gaus filter - Salt n Pepper - sum: 3%")
'''

'''
x = [3,5,7]
y = [21.4767 , 24.3146 , 19.7623]
plt.title("Gaus filter - Salt n Pepper - sum: 5%")
'''
'''
x = [3,5,7]
y = [20.0235 , 22.3123 , 18.9168]
plt.title("Gaus filter - Salt n Pepper - sum: 10%")
'''
'''
x = [3,5,7]
y = [21.5402 , 24.4336 , 19.7221]
plt.title("Gaus filter - Gaus Sum - jacina suma: 1000")
'''
'''
x = [3,5,7]
y = [20.3228 , 22.7207 , 19.0664]
plt.title("Gaus filter - Gaus Sum - jacina suma: 2000")
'''
'''
x = [3,5,7]
y = [19.4273 , 21.5945 , 18.5426]
plt.title("Gaus filter - Gaus Sum - jacina suma: 3000")
'''


plt.ylabel("PSNR")
plt.xlabel("Velicina kernela")
plt.plot(x,y,'ro')
x1 = [0]
y1 = [0]
plt.plot(x1,y1,"wo")
plt.show()

