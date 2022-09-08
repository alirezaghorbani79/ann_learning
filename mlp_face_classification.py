"""
Classification ORL faces dataset using multilayer perceptron. In this program,
i first divided the data into train and test parts and started training on the 
train data. After that, i test the network using test data and show the results.
"""

import numpy as np
import torch
import os
import glob
import cv2
from sklearn.model_selection import train_test_split
from util import to_categorical, calculate_hog


def load_image_from_folder(PATH):
    labels = []
    folders = []
    for it in os.scandir(PATH):
        if it.is_dir():
            path = it.path
            folders.append(path)
            labels.append(int(path.split('s')[-1]) - 1)

    files_train = []
    files_test = []
    for folder in folders:
        files = []
        files.extend(glob.glob(folder+'/*.pgm'))
        paths_train, paths_test, _, _ = train_test_split(
            files, np.zeros(np.array(files).shape), test_size=0.4)
        files_train.append(paths_train)
        files_test.append(paths_test)

    return labels, files_train, files_test


def generate_data():
    PATH = './orl/'

    labels, files_train, files_test = load_image_from_folder(PATH)
    labels_flat = []
    labels_flat_test = []
    for label in labels:
        for _ in range(len(files_train[0])):
            labels_flat.append(label)
        for _ in range(len(files_test[0])):
            labels_flat_test.append(label)

    files_train = np.array(files_train).flatten()
    files_test = np.array(files_test).flatten()

    np.random.seed(13)
    np.random.shuffle(files_train)
    np.random.seed(13)
    np.random.shuffle(labels_flat)

    train_images = []
    for file in files_train:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        train_images.append(img)

    test_images = []
    for file in files_test:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        test_images.append(img)

    X_train = calculate_hog(train_images, 64, 8, (16, 16), (1, 1))
    X_test = calculate_hog(test_images, 64, 8, (16, 16), (1, 1))

    labels_train = to_categorical(labels_flat, 40)
    labels_test = to_categorical(labels_flat_test, 40)
    
    return X_train, X_test, labels_train, labels_test


class Net(torch.nn.Module):
    def __init__(self, input_neurons=128, h1=30, h2=30, output_neurons=40):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_neurons, h1)
        self.fc2 = torch.nn.Linear(h1, h2)
        self.fc3 = torch.nn.Linear(h2, output_neurons)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


def train(X_train, Y_train, X_test, Y_test ,model, optimizer, loss_func, epochs=1000):
    for epoch in range(epochs):
        for inputs, labels in zip(X_train, Y_train):
            prediction = model(inputs.float())
            loss = loss_func(prediction, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            prediction = model(X_test.float())
            loss = loss_func(prediction, Y_test.float())
            print('\repoch ', str(epoch) , ' | loss ' , loss.data.numpy(), end='')


def test(X_test, Y_test, model):
    accuracy = 0
    print()
    for x, y in zip(X_test, Y_test):
        prediction = model(x.float())
        res = torch.argmax(prediction)
        y_res = torch.argmax(y)
        if y_res == res:
            accuracy += 1
        print(f'{y_res} predicted {res}')

    accuracy /= len(X_test)
    print("=======================")
    print(f'accuracy = {accuracy * 100} %')
        

def main():
    X_train, X_test, Y_train, Y_test = generate_data()

    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_func = torch.nn.MSELoss()

    train(torch.tensor(X_train), torch.tensor(Y_train), torch.tensor(X_test), torch.tensor(Y_test), model, optimizer, loss_func, epochs=500)
    test(torch.tensor(X_test), torch.tensor(Y_test), model)


if __name__ == '__main__':
    main()