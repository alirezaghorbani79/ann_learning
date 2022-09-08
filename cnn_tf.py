from traceback import print_tb
from keras import layers, models
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

model = models.Sequential()

model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

WIDTH = 8
HEIGHT = 8

def load_image_from_folder(path):
    images = []
    labels = []
    for filename in os.listdir(path):
        label = filename.split('_')[0]
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        if img is not None and img.shape[0] > WIDTH and img.shape[1] > HEIGHT:
            labels.append(int(label))
            images.append(img)

    return images, labels

def generate_data():
    PATH = './bmp'

    images, labels = load_image_from_folder(PATH)
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.4)

    X_train = np.array(X_train)
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_train = X_train.astype('float32') / 255

    X_test = np.array(X_test)
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
    X_test = X_test.astype('float32') / 255
    
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    return X_train, X_test, Y_train, Y_test



def main():
    X_train, X_test, Y_train, Y_test = generate_data()

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=10, batch_size=32)

    test_loss, test_acc = model.evaluate(X_test, Y_test)

    print(test_loss, test_acc)



if __name__ == "__main__":
    main()