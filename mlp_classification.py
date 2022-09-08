import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from activation_function import sigmoid
from util import classify, to_categorical


class OneLayerMlp():
    def train(self, X, Y, h1=5, eta=0.02, epochs=200):
        w1 = 2 * np.random.rand(X.shape[1], h1) - 1
        b1 = 2 *  np.random.rand(h1) - 1

        w_out = 2 * np.random.rand(h1, 5) - 1
        b_out = 2 * np.random.rand(5) - 1

        for epoch in range(epochs):
            loss = 0
            predicted_points = []
            for inputs, label in zip(X, Y):
                o1 = sigmoid(np.dot(inputs, w1) + b1)
                o2 = sigmoid(np.dot(o1, w_out) + b_out)

                index, predicted = classify(o2, 5)
                predicted_points.append(index)

                delta_out = 2 * (np.subtract(o2, label) * sigmoid(o2, der=True))
                delta_1 = 2 * (np.dot(w_out, delta_out) * sigmoid(o1, der=True))

                w_out = np.subtract(w_out, (eta * np.kron(o1, delta_out).reshape(w_out.shape)))
                b_out = np.subtract(b_out, (eta * delta_out))

                w1 = np.subtract(w1, (eta * np.kron(inputs, delta_1).reshape(w1.shape)))
                b1 = np.subtract(b1, (eta * delta_1))

                loss += np.sum(abs(np.subtract(predicted, label))) / 2

            loss /= X.shape[0]

        
            plt.cla()
            plt.ylim((-1.75, 1.75))
            plt.xlim((-2.2, 2.2))
            plt.scatter(X[:, 0], X[:, 1], c=predicted_points, alpha=0.5)
            plt.text(0.2, 1.5, 'epoch = {:2d}| loss = {:.3f}'.format(epoch, loss), fontdict={'size': 12, 'color':  'blue'})
            plt.pause(0.001)

            if (loss < 0.01):
                break


def main():
    X, y = datasets.make_blobs(n_samples=250, centers=[(-0.8, -1), (-1.5, 0.25), (0, 1), (1.5, 0.25), (0.8, -1)], cluster_std=0.15)
    
    labels = to_categorical(y, 5)

    one_layer_mlp = OneLayerMlp()
    one_layer_mlp.train(X, labels, h1=5, eta=0.1, epochs=500)
    plt.show()


if __name__ == "__main__":
    main()