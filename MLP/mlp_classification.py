import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from util import sigmoid


class One_layer_mlp():
    def __init__(self):
        pass

    def classify(self, arr):
        chosen = np.where(arr == np.amax(arr))[0][0]
        pridicted = [0, 0, 0, 0, 0]
        pridicted[chosen] = 1
        return chosen, pridicted

    def train(self, X, Y, h1=5, eta=0.02, epochs=200):
        w1 = 2 * np.random.rand(X.shape[1], h1) - 1
        b1 = 2 *  np.random.rand(h1) - 1

        w_out = 2 * np.random.rand(h1, 5) - 1
        b_out = 2 * np.random.rand(5) - 1

        loss = 0

        for epoch in range(epochs):
            print('epoch ' + str(epoch))
            print('loss ' + str(loss))
            loss = 0
            pridicted_points = []
            for inputs, label in zip(X, Y):
                o1 = sigmoid(np.dot(inputs, w1) + b1)
                o2 = sigmoid(np.dot(o1, w_out) + b_out)

                chosen, pridicted = self.classify(o2)
                pridicted_points.append(chosen)

                delta_out = 2 * (np.subtract(o2, label) * sigmoid(o2, der=True))
                delta_1 = 2 * (np.dot(w_out, delta_out) * sigmoid(o1, der=True))

                w_out = np.subtract(w_out, (eta * np.kron(o1, delta_out).reshape(w_out.shape)))
                b_out = np.subtract(b_out, (eta * delta_out))

                w1 = np.subtract(w1, (eta * np.kron(inputs, delta_1).reshape(w1.shape)))
                b1 = np.subtract(b1, (eta * delta_1))

                loss += np.sum(abs(np.subtract(pridicted, label))) / 2

            loss /= X.shape[0]

        
            plt.cla()
            plt.scatter(X[:, 0], X[:, 1], c=pridicted_points, alpha=0.5)
            plt.pause(0.001)

            if (loss < 0.01):
                break


def main():
    X, y = datasets.make_blobs(n_samples=250, centers=[(-0.8, -1), (-1.5, 0.25), (0, 1), (1.5, 0.25), (0.8, -1)], cluster_std=0.20, shuffle=True)
    targets = []
    for i in range(len(y)):
        target = [0, 0, 0, 0, 0]
        target[y[i]] = 1
        targets.append(target)

    one_layer_mlp = One_layer_mlp()
    one_layer_mlp.train(X, targets, h1=5, eta=0.1, epochs=500)
    plt.show()


if __name__ == "__main__":
    main()