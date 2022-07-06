import torch
import numpy as np
import matplotlib.pyplot as plt
from util import linear_act, sigmoid_act, tanh_act


class One_layer_mlp():
    def __init__(self):
        pass


    def train(self, X, Y, h1=5, eta=0.1, epochs=200):
        w1 = 2 * (np.random.rand(h1, X.shape[1]) - 0.5)
        b1 = 2 * (np.random.rand(h1) - 0.5)

        w_out = 2 * (np.random.rand(h1) - 0.5)
        b_out = 2 * (np.random.rand(1) - 0.5)

        loss = 0

        for epoch in range(epochs):
            loss = 0
            print('epoch ' + str(epoch))
            for I in range(0, X.shape[0] - 1):
                x = X[I]
                y = Y[I]

                z1 = tanh_act(np.dot(w1, x) + b1)
                output = tanh_act(np.dot(w_out, z1) + b_out)

                delta_out = 2 * (output - y) * tanh_act(output, der=True)
                delta_1 = delta_out * w_out * tanh_act(z1, der=True)

                w_out = w_out - eta * delta_out * z1
                b_out = b_out - eta * delta_out

                w1 = w1 - eta * np.dot(delta_1.reshape(-1, 1), x.reshape(1, -1))
                b1 = b1 - eta* delta_1

                loss += (output - y) ** 2

            loss /= X.shape[0]

            if (epoch % 10 == 0):
                z1 = tanh_act(np.dot(w1, X.T) + b1.reshape(b1.shape[0], 1))
                prediction = tanh_act(np.dot(w_out, z1) + b_out)
                plt.cla()
                plt.axis([-1, 9, -1.5, 1.5])
                plt.scatter(X, Y, color = "blue", alpha=0.2)
                plt.scatter(X, prediction, color='green', alpha=0.5)
                plt.text(6.0, -1.0, 'Epoch = %d' % epoch, fontdict={'size': 14, 'color':  'red'})
                plt.text(6.0, -1.25, 'Loss = %.4f' % loss, fontdict={'size': 14, 'color':  'red'})                
                plt.pause(0.01)
            if(loss < 0.001):
                break

        return w1, b1, w_out, b_out, loss


x = torch.linspace(0, 8, 9) 
x = torch.unsqueeze(x, dim=1)
y = torch.tensor([0, 0.84, 0.91, 0.14, -0.77, -0.96, -0.28, 0.66, 0.99])

one_layer_mlp = One_layer_mlp()

w1, b1, w_out, b_out, loss = one_layer_mlp.train(x.detach().numpy(), y.detach().numpy(), h1=10, eta=0.01, epochs=2500)

plt.waitforbuttonpress()
