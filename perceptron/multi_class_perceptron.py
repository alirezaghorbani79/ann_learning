import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

class Perceptron():
    def __init__(self, learing_rate=0.01, input_neurons = 2, output_neurons = 4):
        self.w =  np.random.rand(input_neurons + 1, output_neurons) - 0.5
        self.learning_rate = learing_rate

    def predict(self, inputs):
        summation = np.dot(inputs, self.w[1:]) + self.w[0]
        activation = (summation > 0.5)

        return activation

    def train(self, X, y, epochs = 100):
        for _ in range(epochs):
            fail_count = 0
            i = 0

            for inputs, label in zip(X, y):
                i = i + 1
                prediction = self.predict(inputs)

                if (np.sum(np.abs(label - prediction)) != 0):
                    self.w[1:] += self.learning_rate * (label - prediction) * inputs.reshape(inputs.shape[0],1)  
                    self.w[0] += self.learning_rate * (label - prediction)
                    fail_count += 1

            if (fail_count == 0):
                break



def plot_decision_boundary(model):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    step = 0.1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

    points = np.c_[xx.ravel(), yy.ravel()]

    predicted_points = []
    for point in points:
        predicted_point = model.predict(point)
        predicted_points.append(predicted_point)

    Z = []
    for p in predicted_points:
        if np.sum(p) == 1:
            Z.append(np.where(p == 1)[0][0])
        else:
            Z.append(5) 

    Z = np.reshape(Z, xx.shape)
 
    plt.figure()
    plt.title("Decision Boundary")
    plt.contourf(xx, yy, Z, cmap='Spectral')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors = 'gray', cmap='Spectral')
    plt.show()


X, y = datasets.make_blobs(n_samples=200, centers=[(-2, -2), (2, 2), (-2, 2), (2, -2)], cluster_std=0.5, shuffle=True)
targets = []
for i in range(len(y)):
    target = [0, 0, 0, 0]
    target[y[i]] = 1
    targets.append(target)

perceptron = Perceptron(0.01, 2, 4)
perceptron.train(X, targets, 100)
plot_decision_boundary(perceptron)


