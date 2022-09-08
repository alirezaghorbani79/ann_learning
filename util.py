import os
import cv2
import numpy as np
from skimage.feature import hog
import math


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


def calculate_block_mean(images):
    WIDTH = 8
    HEIGHT = 8

    image_list = []
    for img in images:
        w = img.shape[1]
        h = img.shape[0]       

        w_step = w / WIDTH
        h_step = h / HEIGHT

        new_img = []

        for i in range(0, h):
            arr = []
            for j in range(0, WIDTH):
                temp_part = img[i][math.floor(j * w_step):math.floor((j + 1) * w_step)]
                arr.append((sum(temp_part) / len(temp_part)) / 255)
            new_img.append(arr)
        
        new_img = np.asarray(new_img).T


        new_image = []
        for i in range(0, new_img.shape[0]):
            arr = []
            for j in range(0, WIDTH):
                temp_part = new_img[i][math.floor(j * h_step):math.floor((j + 1) * h_step)]
                arr.append((sum(temp_part) / len(temp_part)))
            new_image.append(arr)
        

        image_list.append(np.asarray(new_image).T.flatten())

    return image_list

def calculate_hog(images, size=8, orientations=4, pixels_per_cell=(2, 2), cells_per_block=(1, 1)):
    image_list = []

    for image in images:
        img = cv2.resize(image, (size, size))
        hog_image = hog(img, orientations, pixels_per_cell, cells_per_block)
        image_list.append(hog_image)
        
    return image_list

def to_categorical(input, num_classes):
    labels = []
    for el in input:
        label = [0 for _ in range(num_classes)]
        label[el] = 1
        labels.append(label)

    return labels

def classify(prediction, num_classes):
    chosen = np.where(prediction == np.amax(prediction))[0][0]
    predicted = [0 for _ in range(num_classes)]
    predicted[chosen] = 1
    return chosen, predicted
