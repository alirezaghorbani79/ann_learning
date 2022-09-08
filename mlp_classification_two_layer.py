import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from activation_function import sigmoid
from util import classify, to_categorical

class TwoLayerMlp():
    def train(self, X, Y, h1=5, h2=5, output_neurons=5, eta=0.02, epochs=200):
        w1 = 2 * np.random.rand(X.shape[1], h1) - 1
        b1 = 2 *  np.random.rand(h1) - 1

        w2 = 2 * np.random.rand(h1, h2) - 1
        b2 = 2 * np.random.rand(h2) - 1

        w3 = 2 * np.random.rand(h2, output_neurons) - 1
        b3 = 2 * np.random.rand(output_neurons) - 1

        for epoch in range(epochs):
            loss = 0
            predicted_points = []
            for inputs, label in zip(X, Y):
                o1 = sigmoid(np.dot(inputs, w1) + b1)
                o2 = sigmoid(np.dot(o1, w2) + b2)
                o3 = sigmoid(np.dot(o2, w3) + b3)

                chosen, predicted = classify(o3, 5)
                predicted_points.append(chosen)

                delta3 = 2 * (np.subtract(o3, label) * sigmoid(o3, der=True))
                delta2 = 2 * (np.dot(w3, delta3) * sigmoid(o2, der=True))
                delta1 = 2 * (np.dot(w2, delta2) * sigmoid(o1, der=True))

                w3 = np.subtract(w3, (eta * np.kron(o2, delta3).reshape(w3.shape)))
                b3 = np.subtract(b3, (eta * delta3))

                w2 = np.subtract(w2, (eta * np.kron(o1, delta2).reshape(w2.shape)))
                b2 = np.subtract(b2, (eta * delta2))

                w1 = np.subtract(w1, (eta * np.kron(inputs, delta1).reshape(w1.shape)))
                b1 = np.subtract(b1, (eta * delta1))

                loss += np.sum(abs(np.subtract(predicted, label))) / 2

            loss /= X.shape[0]
        
            plt.cla()
            plt.ylim((-1.75, 1.75))
            plt.xlim((-2.2, 2.2))
            plt.scatter(X[:, 0], X[:, 1], c=predicted_points, alpha=0.5)
            plt.text(0.2, 1.5, 'epoch = {:2d}| loss = {:.3f}'.format(epoch, loss), fontdict={'size': 12, 'color':  'blue'})
            plt.pause(0.001)

            if (loss < 0.02):
                break


def main():
    X, y = datasets.make_blobs(n_samples=250, centers=[(-0.8, -1), (-1.5, 0.25), (0, 1), (1.5, 0.25), (0.8, -1)], cluster_std=0.15)
    
    labels = to_categorical(y, 5)

    two_layer_mlp = TwoLayerMlp()
    two_layer_mlp.train(X, labels, h1=5, h2=5, output_neurons=5, eta=0.1, epochs=500)
    plt.show()


if __name__ == "__main__":
    main()