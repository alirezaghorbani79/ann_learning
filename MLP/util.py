import numpy as np

def sigmoid_act(x, der=False):
    f = 1 / (1 + np.exp(-x))

    if (der == True):
        f = f * (1 -f)

    return f

def tanh_act(x, der=False):
    f = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    if (der == True):
        f = 1 - f ** 2

    return f

def linear_act(x, der=False):
    f = x

    if (der == True):
        f = 1

    return f

def sigmoid(x, der=False):
    res = []
    for i in range(len(x)):
        res.append(1 / (1+np.exp(-x[i])))
        if der: res[i] = res[i] * (1-res[i])


    return res