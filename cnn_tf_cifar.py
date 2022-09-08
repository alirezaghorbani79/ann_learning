from keras import layers, models
import numpy as np
import cv2
import os
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
labels_list = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

def load_image_from_folder(path):
    images = []
    labels = []
    for filename in os.listdir(path):
        label = labels_list[filename.split('_')[1][:-4]]
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        labels.append(label)
        images.append(img)

    return images, labels

def generate_data():
    TRAIN_PATH = 'D:\\Programming\\NN\\ann_learning\\cifar\\train'
    TEST_PATH = 'D:\\Programming\\NN\\ann_learning\\cifar\\test'
    # TRAIN_PATH = 'C:\\cifar\\train'
    # TEST_PATH = 'C:\\cifar\\test'
            
    X_train, Y_train = load_image_from_folder(TRAIN_PATH)
    X_test, Y_test = load_image_from_folder(TEST_PATH)

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
    model.fit(X_train, Y_train, epochs=10, batch_size=100)

    test_loss, test_acc = model.evaluate(X_test, Y_test)

    print(test_loss, test_acc)



if __name__ == "__main__":
    main()