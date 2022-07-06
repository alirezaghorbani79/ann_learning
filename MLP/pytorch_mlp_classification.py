import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


class Pytorch_mlp():
    def __init__(self, input_neurons=2, h1=5, h2=5, output_neurons=5, learning_rate=0.1):
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


    def classify(self, arr):
        chosen = np.where(arr == np.amax(arr))[0][0]
        predicted = [0, 0, 0, 0, 0]
        predicted[chosen] = 1
        return chosen, predicted


    def train(self, X, Y, epochs):


        for epoch in range(epochs):
            print("epoch ", epoch)

            pridicted_points = []
            for inputs, label in zip(X, Y):
                prediction = self.network(inputs.float())
                chosen, predicted = self.classify(prediction.data.numpy())
                pridicted_points.append(chosen)

                loss = self.loss_func(prediction, label.float())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            plt.cla()
            plt.scatter(X[:, 0], X[:, 1], c=pridicted_points, alpha=0.5)
            plt.pause(0.001)



def main():
    X, y = datasets.make_blobs(n_samples=250, centers=[(-0.8, -1), (-1.5, 0.25), (0, 1), (1.5, 0.25), (0.8, -1)], cluster_std=0.20, shuffle=True)
    targets = []
    for i in range(len(y)):
        target = [0, 0, 0, 0, 0]
        target[y[i]] = 1
        targets.append(target)

    
    pytorch_mlp = Pytorch_mlp(input_neurons=2, h1=5, h2=5, output_neurons=5, learning_rate=0.1)
    pytorch_mlp.train(torch.from_numpy(X), torch.tensor(targets), 100)
    plt.show()


if __name__ == "__main__":
    main()
