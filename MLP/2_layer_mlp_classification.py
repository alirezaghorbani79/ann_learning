import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from util import sigmoid

class Two_layer_mlp():
    def classify(self, arr):
        chosen = np.where(arr == np.amax(arr))[0][0]
        predicted = [0, 0, 0, 0, 0]
        predicted[chosen] = 1
        return chosen, predicted

    def train(self, X, Y, h1=5, h2=5, output_neurons=5, eta=0.02, epochs=200):
        w1 = 2 * np.random.rand(X.shape[1], h1) - 1
        b1 = 2 *  np.random.rand(h1) - 1

        w2 = 2 * np.random.rand(h1, h2) - 1
        b2 = 2 * np.random.rand(h2) - 1

        w3 = 2 * np.random.rand(h2, output_neurons) - 1
        b3 = 2 * np.random.rand(output_neurons) - 1

        loss = 0

        for epoch in range(epochs):
            print('epoch ' + str(epoch))
            print('loss ' + str(loss))
            loss = 0
            predicted_points = []
            for inputs, label in zip(X, Y):
                o1 = sigmoid(np.dot(inputs, w1) + b1)
                o2 = sigmoid(np.dot(o1, w2) + b2)
                o3 = sigmoid(np.dot(o2, w3) + b3)

                chosen, predicted = self.classify(o3)
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
            plt.scatter(X[:, 0], X[:, 1], c=predicted_points, alpha=0.5)
            plt.pause(0.001)

            if (loss < 0.02):
                break


def main():
    X, y = datasets.make_blobs(n_samples=250, centers=[(-0.8, -1), (-1.5, 0.25), (0, 1), (1.5, 0.25), (0.8, -1)], cluster_std=0.20, shuffle=True)
    targets = []
    for i in range(len(y)):
        target = [0, 0, 0, 0, 0]
        target[y[i]] = 1
        targets.append(target)

    two_layer_mlp = Two_layer_mlp()
    two_layer_mlp.train(X, targets, h1=5, h2=5, output_neurons=5, eta=0.1, epochs=500)
    plt.show()


if __name__ == "__main__":
    main()