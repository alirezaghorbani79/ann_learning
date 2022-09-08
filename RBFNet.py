import torch
import numpy as np
import rbf.torch_rbf as rbf

class RBFNet(torch.nn.Module):
    def __init__(self, input_neurons, output_neurons, layer_centers, centers, basis_func):
        super(RBFNet, self).__init__()
        self.fc1 = rbf.RBF(input_neurons, layer_centers, basis_func, centers)
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

            optimizer.zero_grad()
            prediction = self.forward(X)           
            loss = loss_func(prediction.squeeze(1), Y)
            current_loss += loss.item() - current_loss
            loss.backward()
            optimizer.step()
            print(f'\repoch {epoch} || loss = {current_loss:.6f}', end='')
