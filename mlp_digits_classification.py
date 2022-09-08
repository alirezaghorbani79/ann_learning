import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import os
from util import classify, calculate_block_mean, to_categorical
from activation_function import tanh, sigmoid

WIDTH = 8
HEIGHT = 8


def load_image_from_folder(path):
    images = []
    labels = []
    for filename in os.listdir(path):
        label = filename.split('_')[0]
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None and img.shape[0] > WIDTH and img.shape[1] > HEIGHT:
            labels.append(int(label))
            images.append(img)

    return images, labels


def generate_data():
    PATH = './bmp/'

    images, labels = load_image_from_folder(PATH)
    images = calculate_block_mean(images)

    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.4)

    return X_train, X_test, Y_train, Y_test


class OneLayerMlp():
    def train(self, X_train, Y_train, X_test, Y_test, h1=5, learning_rate=0.02, epochs=200):
        w1 = 2 * np.random.rand(X_train.shape[1], h1) - 1
        b1 = 2 *  np.random.rand(h1) - 1

        w_out = 2 * np.random.rand(h1, 10) - 1
        b_out = 2 * np.random.rand(10) - 1

        loss = 0
        train_loss = []
        test_loss = []
        for epoch in range(epochs):
            loss = 0
            pridicted_points = []
            for inputs, label in zip(X_train, Y_train):
                o1 = tanh(np.dot(inputs, w1) + b1)
                o2 = np.dot(o1, w_out) + b_out

                chosen, pridicted = classify(o2, 10)
                pridicted_points.append(chosen)

                delta_out = 2 * (np.subtract(o2, label) * sigmoid(o2, der=True))
                delta_1 = 2 * (np.dot(w_out, delta_out) * sigmoid(o1, der=True))

                w_out = np.subtract(w_out, (learning_rate * np.kron(o1, delta_out).reshape(w_out.shape)))
                b_out = np.subtract(b_out, (learning_rate * delta_out))

                w1 = np.subtract(w1, (learning_rate * np.kron(inputs, delta_1).reshape(w1.shape)))
                b1 = np.subtract(b1, (learning_rate * delta_1))

                loss += np.sum(abs(np.subtract(pridicted, label))) / 2

            loss /= X_train.shape[0]

            if epoch % 10 == 0:
                train_loss.append(loss)
                o1 = tanh(np.dot(inputs.T, w1) + b1.reshape(b1.shape[0], 1))
                o2 = np.dot(o1, w_out) + b_out
                
                #TODO: calculate loss and plot RMSloss

                x = range(0, epoch + 2, 10)
                plt.cla()
                plt.plot(x, train_loss, color='red', label='train loss', alpha=0.3, marker='o')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.pause(0.01)

            if (loss < 0.01):
                break

def main():
    X_train, X_test, Y_train, Y_test = generate_data()

    labels_train = to_categorical(Y_train, 10)
    labels_test = to_categorical(Y_test, 10)

        
    one_layer_mlp = OneLayerMlp()
    one_layer_mlp.train(np.array(X_train), np.array(labels_train), np.array(X_test), np.array(labels_test), h1=30, learning_rate=0.1, epochs=300)


if __name__ == '__main__':
    main()