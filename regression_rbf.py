from __future__ import barry_as_FLUFL
import torch
from torch.utils.data import Dataset, DataLoader
import rbf_layer as rbf
import numpy as np
import matplotlib.pyplot as plt


class RBFNet(torch.nn.Module):
    def __init__(self, input_neurons, output_neurons, layer_centers, basis_func):
        super(RBFNet, self).__init__()
        self.fc1 = rbf.RBF(input_neurons, layer_centers, basis_func)
        self.fc2 = torch.nn.Linear(layer_centers, output_neurons)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def fit(self, X, Y, epochs, batch_size, learning_rate, loss_func):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            current_loss = 0

            for i in range(len(X) // batch_size):
                x = X[i * batch_size:(i + 1) * batch_size]
                y = Y[i * batch_size:(i + 1) * batch_size]
                optimizer.zero_grad()
                prediction = self.forward(x)
                loss = loss_func(prediction, y)
                current_loss += loss.item()
                loss.backward()
                optimizer.step()


            if epoch % 20 == 0:
                current_loss /= batch_size
                print(f'\repoch {epoch} || loss = {current_loss}', end='')



NUM_SAMPLES = 150

xmin = -4; xmax = 4
X = np.arange(xmin, xmax, (xmax - xmin) / NUM_SAMPLES)
noise = np.random.uniform(-0.1, 0.1, NUM_SAMPLES)
Y = (1 - 4 * X - X ** 3 / 17) * np.sin(X ** 2)

X = torch.from_numpy(X.reshape(-1,1)).float()
Y = torch.from_numpy(Y.reshape(-1,1)).float()

rbfnet = RBFNet(1, 1, 100, rbf.gaussian)
rbfnet.fit(X, Y, 1000, 50, 0.01, torch.nn.MSELoss())
rbfnet.eval()

with torch.no_grad():
    preds = rbfnet(X).data.numpy()

plt.figure()
plt.plot(X, Y, label='Predicted', c='blue', alpha=0.5)
plt.scatter(X, preds, label='Predicted', s=10, c='red', alpha=0.5)
plt.show()