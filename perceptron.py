import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

def predict(inputs, weights):
    summation = np.dot(inputs, weights[1:]) + weights[0]
    activation = 1.0 if (summation > 0.0) else 0.0

    return activation


x, y = datasets.make_blobs(n_samples=200, centers=[(-1, -1), (1, 1)], cluster_std=0.5)

plt.scatter(x[:, 0], x[:, 1], c=y, cmap='jet_r')

num_of_inputs = 2
epochs = 100
learning_rate = 0.01
w = np.random.random(num_of_inputs + 1) - 0.5

for epoch in range(epochs):
    fail_count = 0
    i = 0

    for inputs, label in zip(x, y):
        i = i + 1
        prediction = predict(inputs, w)

        if (label != prediction):
            w[1:] += learning_rate * (label - prediction) * inputs.reshape(inputs.shape[0])
            w[0] += learning_rate * (label - prediction)
            fail_count += 1

            plt.cla()
            plt.scatter(x[:, 0], x[:, 1], c=y, cmap='jet_r')
            line_x = np.arange(-3, 3, 0.1)
            line_y = (-w[0] - w[1] * line_x) / w[2]
            plt.plot(line_x, line_y)
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)
            plt.text(0, -2.7, 'epoch|iter = {:2d}|{:2d}'.format(epoch, i), fontdict={'size': 14, 'color':  'blue'})
            plt.pause(0.01)

    if (fail_count == 0):
        plt.show()
        break