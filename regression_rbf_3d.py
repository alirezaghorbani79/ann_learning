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

    def fit(self, X, Y, epochs, learning_rate, loss_func):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            optimizer.zero_grad()
            prediction = self.forward(X)           
            loss = loss_func(prediction.squeeze(1), Y)
            loss.backward()
            optimizer.step()
            print(f'\repoch {epoch} || loss = {loss.item():.6f}', end='')


def main():
    NUM_SAMPLES = 50

    min = -4; max = 4
    X_lin = Y_lin = np.arange(min, max, (max - min) / NUM_SAMPLES)
    noise = np.random.uniform(-0.1, 0.1, NUM_SAMPLES)
    X, Y = np.meshgrid(X_lin, Y_lin)
    Z = np.sin(X) * np.cos(Y)
    X_train = np.expand_dims(X.flatten(), axis=1)
    Y_train = np.expand_dims(Y.flatten(), axis=1)
    train_data = np.concatenate((X_train, Y_train), axis=1)

    rbfnet = RBFNet(2, 1, 50, rbf.gaussian)
    rbfnet.fit(torch.from_numpy(train_data).float(), torch.from_numpy(Z.flatten()).float(), 1000, 0.02, torch.nn.MSELoss())
    rbfnet.eval()

    with torch.no_grad():
        prediction = rbfnet(torch.from_numpy(train_data).float()).data.numpy()

    fig = plt.figure(figsize = (12,6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X, Y, Z, cmap = plt.cm.cividis)
    ax.set_title('Main')

    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(X, Y, prediction.reshape(Z.shape), cmap = plt.cm.cividis)
    ax.set_title('Predicted')

    ax.set_xlabel('x', labelpad=20)
    ax.set_ylabel('y', labelpad=20)
    ax.set_zlabel('z', labelpad=20)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()