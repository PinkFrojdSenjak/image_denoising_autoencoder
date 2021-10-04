import numpy as np
from math import e
import random
import math

np.random.seed(seed = 0)
random.seed(a = 0)

def sigmoid(x):
    return 1 / (1 + e ** (-x))

def sigmoid_der(x):
    return  sigmoid(x) * ( 1 - sigmoid(x))

def flatten(a):
    b = np.array([])
    for i in a:
        b = np.concatenate((b,i), axis = None)
    return b

def shift(a):
    for i in range(len(a)-1):
        a[i] = a[i+1]
    return a
    

class lbfgs_parametars:
    def __init__(self, m=6,  epsilon=1e-8, c1=1e-4, c2=0.9, max_linesearch=10, min_step=1e-20, max_step=1e20):
        self.m = m
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.max_linesearch = max_linesearch
        self.min_step = min_step
        self.max_step = max_step


class Net:
    def __init__(self, layer_sizes):
        self.num_layers = len(layer_sizes)
        self.biases = [np.random.rand(y, 1) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.lbfgs_parametars = lbfgs_parametars()
        self.num_biases = np.sum(layer_sizes[1:])
        self.num_weights = 0
        for i in range(len(layer_sizes[:-1])):
            self.num_weights += layer_sizes[i] * layer_sizes[i+1]
        self.n = self.num_biases + self.num_weights
        self.s = np.zeros((self.lbfgs_parametars.m,self.n), dtype = float)
        self.y = np.zeros((self.lbfgs_parametars.m,self.n), dtype = float)
        self.k = 1
        self.step = 1.0
        self.d = 0
        self.change = 1
        

    def feedFoward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def fit_model(self, training_data, test_data, epochs, mini_batch_size):
        if test_data: 
            n_test = len(test_data)

        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[p:p+mini_batch_size] for p in range(0, n, mini_batch_size)]
            print(len(mini_batches))
            for mini_batch in mini_batches:
                self.update(mini_batch)
            if test_data:
                print ("Epoch "+str(j)+": "+str(self.evaluate(test_data, n_test))+" /"+str(n_test))
            else:
                print(j)
            

    def update(self, mini_batch):

        x = np.concatenate((flatten(self.biases),  flatten(self.weights)), axis = None)
        #g = self.grad(mini_batch)
        if self.change:
            x , self.change = self.lbfgs(x, self.grad(mini_batch), mini_batch)




    def backprop(self, x, y):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x] 
        zs = [] 
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

    
        delta = 2*(activations[-1] - y) * sigmoid_der(zs[-1])

        grad_b[-1] = delta
        
        grad_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sd = sigmoid_der(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sd
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (grad_b, grad_w)

    def evaluate(self,test_data,n_test):
        test_results = 0.0
        for (x,y) in test_data:
            mse = 0.0
            for i in range(len(y)):
                mse += np.sum( (self.feedFoward(x) - y)**2 ) / len(y)
                test_results += 10 * math.log(1.0/mse, 10)
            test_results /= n_test
            return test_results



    def grad(self,mini_batch):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            d_grad_b, d_grad_w = self.backprop(x, y)
            grad_b = [gb+dgb for gb, dgb in zip(grad_b, d_grad_b)]
            grad_w = [gw+dgw for gw, dgw in zip(grad_w, d_grad_w)]
        
        grad_b = [gb/len(mini_batch) for gb in grad_b]
        grad_w = [gw/len(mini_batch) for gw in grad_w]


        return np.concatenate((flatten(grad_b),  flatten(grad_w)), axis = None)


    #cost funciton
    def f(self,mini_batch):
        cost = 0
        for x,y in mini_batch:
            cost += np.sum( (self.feedFoward(x) - y)**2 )/len(y)
        return cost

    def linesearch(self,mini_batch, fx, x, g, d, step, xp, gp, c1, c2, min_step, max_step, max_linesearch):
        count = 0
        dec = 0.5
        inc = 2.1
        result = {'status':0,'fx':fx,'step':step,'x':x, 'g':g}

        dginit = np.dot(g, d)

        if 0 < dginit:
            result['status'] = -1
            return result

        finit = fx
        dgtest = c1 * dginit

        while True:
            x = xp
            x = x + d * step
            self.change_weights_biases(d*step)
                
            fx = self.f(mini_batch)
            g = self.grad(mini_batch)
            
            count = count + 1
            # Armijo condition
            if fx > finit + (step * dgtest):
                width = dec
            else:
                # check the wolfe condition
                dg = np.dot(g, d)
                if dg < c2 * dginit:
                    width = inc
                else:
                    # check the strong wolfe condition
                    if dg > -c1 * dginit:
                        width = dec
                    else:
                        result = {'status':0, 'fx':fx, 'step':step, 'x':x, 'g':g}
                        return result
            if step < min_step:
                result['status'] = -1
                return result

            if step > max_step:
                result['status'] = -1
                return result
            
            if max_linesearch <= count:
                result = {'status':0, 'fx':fx, 'step':step, 'x':x, 'g':g}
                return result	

            step = step * width



    def lbfgs(self, x,  g, mini_batch):
        m = self.lbfgs_parametars.m

        #g = self.grad(mini_batch)

        fx = self.f(mini_batch)

        alfa = np.zeros(m, dtype = float)

        step = 1.0

        if self.k == 1:
            self.d = -1 * g

            gnorm = np.sqrt(np.dot(g,g))
            xnorm = np.sqrt(np.dot(x,x))

            if xnorm < 1.0:
                xnorm = 1.0
            if gnorm / xnorm <= self.lbfgs_parametars.epsilon:
                print("[INFO] vec minimizovano")
                return x
            step = 1.0 / np.sqrt(np.dot(self.d, self.d))


        xp = np.copy(x)

        gp = np.copy(g)

        fxp = fx

        ls = self.linesearch(mini_batch, fx, x, g, self.d, step, xp, gp, self.lbfgs_parametars.c1, self.lbfgs_parametars.c2,self.lbfgs_parametars.min_step, self.lbfgs_parametars.max_step, self.lbfgs_parametars.max_linesearch)

        if ls['status'] < 0:
            x = np.copy(xp)
            g = np.copy(gp)
            return (x, 0)

        fx = ls['fx']
        step = ls['step']
        x = ls['x']
        g = ls['g']

        if np.sum(abs(g)) < self.lbfgs_parametars.epsilon:
            return (x, 0)
        
        '''xnorm = np.sqrt(np.dot(x, x))
        gnorm = np.sqrt(np.dot(g, g))
        
        if xnorm < 1.0:
            xnorm = 1.0
        if gnorm / xnorm <= self.lbfgs_parametars.epsilon:
            return (x, 0)'''

        br = min(m, self.k) - 1

        if self.k >= m:
            self.s = shift(self.s)
        self.s[br] = x - xp

        if self.k >= m:
            self.y = shift(self.y)
        self.y[br] = g - gp 

        ys = np.dot(self.y[br], self.s[br])
        yy = np.dot(self.y[br], self.y[br])

        self.d = -1 * g

        for i in range(br, 0, -1):

            ys = np.dot(self.y[i], self.s[i])
            if ys < 0 and ys > -0.001:
                alfa[i] = np.dot(self.s[i], self.d) /( -0.000000001 )
            elif ys >= 0 and ys < 0.001:
                alfa[i] = np.dot(self.s[i], self.d) / 0.000000001 
            else:
                alfa[i] = np.dot(self.s[i], self.d) / ys

            self.d -= alfa[i] * self.y[i]

        self.d *= (ys/yy)

        for i in range(0, br):
            ys = np.dot(self.y[i], self.s[i])

            if ys < 0 and ys > -0.001:
                beta = np.dot(self.s[i], self.d) /( -0.000000001 )
            elif ys >= 0 and ys < 0.001:
                beta = np.dot(self.s[i], self.d) / 0.000000001 
            else:
                beta = np.dot(self.s[i], self.d) / ys

            self.d += self.s[i] * (alfa[i] - beta)

        self.k += 1

        return (x, 1)
    
    def change_weights_biases(self, a):
        br = 0
        for i in range(len(self.biases)):
            for j in range(len(self.biases[i])):
                for k in range(len(self.biases[i][j])):
                    self.biases[i][j][k] += a[br]
                    br+=1
        
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] += a[br]
                    br+=1
        


        
        









