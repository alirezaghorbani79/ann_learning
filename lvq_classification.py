"""
Classification of two classes of gaussian points using learning vector quantization.
In this program, i have created two classes of normal gaussian distribution on
four different points and classify them usung LVQ.
"""

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


class LVQ():
    def _get_min_dist_index(self, input, prototypes):
        distances = [np.sum(np.subtract(input, proto) ** 2) for proto in prototypes]
        min_dist_index = np.argmin(distances)  

        return min_dist_index

    def train(self, X, Y, n_prototypes, learning_rate, epochs):
        unique_labels = list(set(Y))
        self.prototypes = np.zeros((n_prototypes * len(unique_labels), X.shape[1]))
        self.proto_labels = unique_labels * n_prototypes

        for i in range(len(unique_labels)):
            init_data = X[Y == i, :][0:n_prototypes]

            for j in range(len(init_data)):
                self.prototypes[i + (j * len(unique_labels))] = init_data[j]

        for epoch in range(epochs):
            for input, label in zip (X, Y):
                min_dist_index = self._get_min_dist_index(input, self.prototypes)

                winner_proto = self.prototypes[min_dist_index]
                winner_label = self.proto_labels[min_dist_index]

                if winner_label == label:
                    sign = 1
                else:
                    sign = -1
                self.prototypes[min_dist_index] = np.add(winner_proto, 
                                                            np.subtract(input, winner_proto)
                                                            * learning_rate * sign)

            plt.cla()
            plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors = 'gray', cmap='Spectral')
            plt.scatter(self.prototypes[:, 0], self.prototypes[:, 1], marker="*", c=self.proto_labels)
            plt.pause(0.1)

        return self.prototypes

    def predict(self, X):
        predicted_labels = []
        for input in X:
            min_dist_index = self._get_min_dist_index(input, self.prototypes)
            predicted_labels.append(self.proto_labels[min_dist_index])

        return predicted_labels


def plot_decision_boundary(X, Y, prototypes,model):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    step = 0.1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

    points = np.c_[xx.ravel(), yy.ravel()]

    predicted_points = model.predict(points)
    predicted_points = np.reshape(predicted_points, xx.shape)
    
    plt.cla()
    plt.title("Decision Boundary")
    plt.contourf(xx, yy, predicted_points, cmap='turbo')
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors = 'gray', cmap='plasma')
    plt.scatter(prototypes[:, 0], prototypes[:, 1], marker="*", cmap='Spectral')

    plt.show()

def main():
    X1, y1 = datasets.make_blobs(n_samples=100, centers=[(-2, -2), (-2, 2)], cluster_std=0.5)
    X2, y2 = datasets.make_blobs(n_samples=100, centers=[(2, 2), (2, -2)], cluster_std=0.5)
    X_train = np.concatenate((X1, X2), axis=0)
    Y_train = np.concatenate((y1, y2), axis=0)
    np.random.seed(13)
    np.random.shuffle(X_train)
    np.random.seed(13)
    np.random.shuffle(Y_train)

    lvq_net = LVQ()
    prototypes = lvq_net.train(X_train, Y_train, 5, 0.1, 10)

    plot_decision_boundary(X_train, Y_train, prototypes, lvq_net)

if __name__ == '__main__':
    main()