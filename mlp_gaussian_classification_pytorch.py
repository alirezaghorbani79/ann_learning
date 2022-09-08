"""
Classification 5 classes of gaussian points using multilayer perceptron
In this example, i have used a PyTorch sequential model to create an MLP network.
Also, i have used a sigmoid for the activation function.
"""

import torch
import matplotlib.pyplot as plt
from sklearn import datasets
from util import classify, to_categorical


class TorchMlp():
    def __init__(self, input_neurons=2, h1=10, h2=10, output_neurons=5, learning_rate=0.1):
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_neurons, h1),
            torch.nn.Sigmoid(),
            torch.nn.Linear(h1, h2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(h2, output_neurons),
            torch.nn.Sigmoid()
        )

        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=learning_rate)
        self.loss_func = torch.nn.MSELoss()

    def train(self, X, Y, epochs):
        for epoch in range(epochs):
            loss = 0
            predicted_points = []

            for inputs, label in zip(X, Y):
                prediction = self.network(inputs.float())
                chosen, predicted = classify(prediction.data.numpy(), 5)
                predicted_points.append(chosen)

                loss = self.loss_func(prediction, label.float())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            plt.cla()
            plt.ylim((-1.75, 1.75))
            plt.xlim((-2.2, 2.2))
            plt.scatter(X[:, 0], X[:, 1], c=predicted_points, alpha=0.5)
            plt.text(0.2, 1.5, 'epoch = {:2d}| loss = {:.3f}'.format(epoch, loss), fontdict={'size': 12, 'color':  'blue'})
            plt.pause(0.001)

def main():
    X, y = datasets.make_blobs(n_samples=250, centers=[(-0.8, -1), (-1.5, 0.25), (0, 1), (1.5, 0.25), (0.8, -1)], cluster_std=0.15)
    
    labels = to_categorical(y, 5)
    
    pytorch_mlp = TorchMlp(input_neurons=2, h1=5, h2=5, output_neurons=5, learning_rate=0.1)
    pytorch_mlp.train(torch.from_numpy(X), torch.tensor(labels), 100)
    plt.show()

if __name__ == "__main__":
    main()
