"""
Activation functions, which are used in the programs.
"""

import numpy as np


def sigmoid(x, der=False):
    f = 1 / (1 + np.exp(-x))

    if (der == True):
        f = f * (1 -f)

    return f

def tanh(x, der=False):
    f = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    if (der == True):
        f = 1 - f ** 2

    return f

def linear(x, der=False):
    f = x

    if (der == True):
        f = 1

    return f

def ReLU(x, der=False):
    if (der == True):
        f = np.heaviside(x, 1)
    else :
        f = np.maximum(x, 0)
    return f
