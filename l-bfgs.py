import numpy as np

#funkcija f(x,y) = cos(x) + 100sin2(y)
def f(x):
    return np.sin(x[0]) + 100*np.sin(x[1])**2

#gradijent funkcije
def grad(x):
    g = np.array([], dtype = float)
    g = np.append(g, np.cos(x[0]))
    g = np.append(g, 200 * np.sin(x[1])*np.cos(x[1]))
    return g

def shift(a):
    for i in range(len(a)-1):
        a[i] = a[i+1]
    return a

#lbfgs parametri
eps = 1e-5
min_step = 1e-20
max_step = 1e20
max_linesearch = 10
c1 = 1e-4
c2 = 0.9
m = 10

def linesearch(fx, x, g, d, step, xp, gp, c1, c2, min_step, max_step, max_linesearch):
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
            
        fx = f(x)
        g = grad(x)
        
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


        
def lbfgs(x, mini_batch, a):
    n = len(x)

    s = np.zeros((m,n), dtype = float)
    y = np.zeros((m,n), dtype = float)

    alfa = np.zeros(m, dtype = float)

    g = grad(x)

    fx = f(x)

    d = -g

    gnorm = np.sqrt(np.dot(g,g))
    xnorm = np.sqrt(np.dot(x,x))

    if xnorm < 1.0:
        xnorm = 1.0
    if gnorm / xnorm <= eps:
        print("[INFO] vec minimizovano")
        return x

    step = 1.0 / np.sqrt(np.dot(d, d))

    k = 1

    while True:
        xp = x.copy()

        gp = g.copy()

        ls = linesearch(fx, x, g, d, step, xp, gp, c1, c2, min_step, max_step, max_linesearch)

        if ls['status'] < 0:
            x = np.copy(xp)
            g = np.copy(gp)
            return x

        fx = ls['fx']
        step = ls['step']
        x = ls['x']
        g = ls['g']
        
        xnorm = np.sqrt(np.dot(x, x))
        gnorm = np.sqrt(np.dot(g, g))
        
        if xnorm < 1.0:
            xnorm = 1.0
        if gnorm / xnorm <= eps:
            return x

        br = min(m, k) - 1

        if k >= m:
            s = shift(s)
        s[br] = x - xp

        if k >= m:
            y = shift(y)
        y[br] = g - gp 

        ys = np.dot(y[br], s[br])
        yy = np.dot(y[br], y[br])

        d = -1 * g

        for i in range(br, 0, -1):
            alfa[i] = np.dot(s[i], d) / np.dot(y[i], s[i])
            d -= alfa[i] * y[i]

        d *= (ys/yy)

        for i in range(0, br):
            beta = np.dot(y[i], d) / np.dot(y[i], s[i])
            d += s[i] * (alfa[i] - beta)

        step = 1.0
        
        k += 1

x = np.array([1,1], dtype = float)
xmin = lbfgs(x)
print(xmin)
print(f(xmin))