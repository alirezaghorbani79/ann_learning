"""
Function regression using multilayer perceptron.
In this example, i have used single-layer perceptron for regression
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from activation_function import tanh


class OneLayerMlp():
    def train(self, X, Y, h1=5, eta=0.1, epochs=200):
        w1 = 2 * (np.random.rand(h1, X.shape[1]) - 0.5)
        b1 = 2 * (np.random.rand(h1) - 0.5)

        w_out = 2 * (np.random.rand(h1) - 0.5)
        b_out = 2 * (np.random.rand(1) - 0.5)

        for epoch in range(epochs):
            loss = 0
            for x, y  in zip(X, Y):
                z1 = tanh(np.dot(w1, x) + b1)
                output = tanh(np.dot(w_out, z1) + b_out)
                
                delta_out = 2 * (output - y) * tanh(output, der=True)
                delta_1 = delta_out * w_out * tanh(z1, der=True)

                w_out = w_out - eta * delta_out * z1
                b_out = b_out - eta * delta_out

                w1 = w1 - eta * np.dot(delta_1.reshape(-1, 1), x.reshape(1, -1))
                b1 = b1 - eta* delta_1

                loss += (output - y) ** 2

            loss /= X.shape[0]

            if (epoch % 10 == 0):
                z1 = tanh(np.dot(w1, X.T) + b1.reshape(b1.shape[0], 1))
                prediction = tanh(np.dot(w_out, z1) + b_out)
                plt.cla()
                plt.ylim([-0.25, 1.25])
                plt.text(0, 1.15, 'epoch = {:2d}| loss = {:.3f}'.format(epoch, loss[0]), fontdict={'size': 12, 'color':  'blue'})
                plt.scatter(X, Y, alpha=0.5)
                plt.plot(X, prediction, 'r-', lw=2)
                plt.pause(0.1)

            if(loss < 0.002):
                break

        plt.waitforbuttonpress()

def input_function(X):
    return 0.5 *(np.sin(2 * np.pi * X * X) + 1.0)

def main():
    x_max = 1.0
    x_min = -1.0

    X = np.arange(x_min, x_max, (x_max - x_min) / 200)
    Y = input_function(X) + 0.2 * np.random.normal(0, 0.1, len(X))

    X = torch.from_numpy(X.reshape(-1,1)).float()
    Y = torch.from_numpy(Y.reshape(-1,1)).float()

    one_layer_mlp = OneLayerMlp()
    one_layer_mlp.train(X.detach().numpy(), Y.detach().numpy(), h1=8, eta=0.1, epochs=2000)

if __name__ == '__main__':
    main()