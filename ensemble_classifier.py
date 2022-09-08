"""
Comparison between different methods of classification.
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn import svm
from matplotlib import pyplot as plt
from util import load_image_from_folder, calculate_block_mean, calculate_hog, to_categorical


def generate_data():
    PATH = './bmp'

    images, labels = load_image_from_folder(PATH)
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.4)

    block_mean_train_x = calculate_block_mean(X_train)
    hog_train_x = calculate_hog(X_train)    
    block_mean_test_x = calculate_block_mean(X_test)
    hog_test_x = calculate_hog(X_test)

    comb_train_x = block_mean_train_x + hog_train_x
    comb_test_x = block_mean_test_x + hog_test_x
    comb_train_y = Y_train + Y_train
    comb_test_y = Y_test + Y_test

    return block_mean_train_x, block_mean_test_x, hog_train_x, hog_test_x, comb_train_x, comb_test_x, comb_train_y, comb_test_y, Y_train, Y_test

class Net(torch.nn.Module):
    def __init__(self, input_neurons=64, h1=30, output_neurons=10):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_neurons, h1)
        self.fc2 = torch.nn.Linear(h1, output_neurons)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)

        return x

def train(X_train, Y_train, model, optimizer, loss_func, epochs=1000):

    BATCH_SIZE = 20

    for epoch in range(epochs):
        for i in range(len(X_train) // BATCH_SIZE):
            images = X_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            labels = Y_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            prediction = model(images.float())
            loss = loss_func(prediction, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            prediction = model(X_train.float())
            loss = loss_func(prediction, Y_train.float())

            print('epoch ', str(epoch) , ' | loss ' , loss.data.numpy())

def test(X_test, Y_test, model, loss_func):
    prediction = model(torch.tensor(X_test).float())
    loss = loss_func(prediction, torch.tensor(Y_test).float())

    return prediction, loss.data.numpy()

def present(title, loss):
    print(f'|| {title} loss = {loss} ||')

def main():
    block_mean_train_x, block_mean_test_x, hog_train_x, hog_test_x, comb_train_x, comb_test_x, comb_train_y, comb_test_y, Y_train, Y_test = generate_data()
    train_labels = to_categorical(Y_train, 10)
    test_labels = to_categorical(Y_test, 10)
    comb_train_labels = to_categorical(comb_train_y, 10)
    comb_test_labels = to_categorical(comb_test_y, 10)
    C = 10

    block_mean_model = Net()
    block_mean_optimizer = torch.optim.SGD(block_mean_model.parameters(), lr=0.1)
    
    hog_model = Net()
    hog_optimizer = torch.optim.SGD(hog_model.parameters(), lr=0.1)

    comb_model = Net()
    comb_optimizer = torch.optim.SGD(comb_model.parameters(), lr=0.1)
    
    SVMnet = svm.SVC(kernel='linear', C=C)

    loss_func = torch.nn.MSELoss()

    #Train phase
    train(torch.tensor(block_mean_train_x), torch.tensor(train_labels), block_mean_model, block_mean_optimizer, loss_func, epochs=1000)
    train(torch.tensor(hog_train_x), torch.tensor(train_labels), hog_model, hog_optimizer, loss_func, epochs=1000)
    train(torch.tensor(comb_train_x), torch.tensor(comb_train_labels), comb_model, comb_optimizer, loss_func, epochs=1000)
    SVMnet.fit(block_mean_train_x, Y_train)

    #Test phase
    block_mean_prediction, block_mean_loss = test(block_mean_test_x, test_labels, block_mean_model, loss_func)
    hog_prediction, hog_loss = test(hog_test_x, test_labels, hog_model, loss_func)
    total_prediction = np.divide(np.add(block_mean_prediction.detach().numpy(), hog_prediction.detach().numpy()), 2)
    chertopert = loss_func(torch.from_numpy(total_prediction), torch.tensor(test_labels).float())
    comb_prediction, comb_loss = test(comb_test_x, comb_test_labels, comb_model, loss_func)
    svm_loss = 1 - SVMnet.score(block_mean_test_x, Y_test)

    present('block mean feature prediction', block_mean_loss)
    present('hog feature prediction', hog_loss)
    present ('combine block mean and hog features predictions', chertopert)
    present('combonential features prediction', comb_loss)
    present('svm network prediction', svm_loss)

if __name__ == '__main__':
    main()