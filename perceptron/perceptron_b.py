import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

def predict(inputs, weights):
    summation = np.dot(inputs, weights[1:]) + weights[0]
    activation = (summation > 0.5)*1.0

    return activation

x, y = datasets.make_blobs(n_samples=200, centers=[(-2, -2), (2, 2), (-2, 2), (2, -2)], cluster_std=0.5, shuffle=True)

targets = []
for i in range(len(y)):
    temp = [0, 0, 0, 0]
    temp[y[i]] = 1
    targets.append(temp)

plt.scatter(x[:, 0], x[:, 1], c=y, cmap='jet_r')


input_neurons = 2
output_neurons = 4





num_of_inputs = 2
epochs = 100
learning_rate = 0.01
w = np.random.rand(input_neurons + 1, output_neurons) - 0.5

for ep in range(epochs):
    fail_count = 0
    i = 0

    for inputs, label in zip(x, targets):
        i = i + 1
        prediction = predict(inputs, w)

        if (np.sum(np.abs(label - prediction)) != 0):
            w[1:] += learning_rate * (label - prediction) * inputs.reshape(inputs.shape[0],1)  
            w[0] += learning_rate * (label - prediction)
            fail_count += 1

    if (fail_count == 0):
        
        break



# class Perceptron():
    
#     def __init__(self, epochs, learning_rate) -> None:
#         self.epochs = epochs
#         self.learning_rate = learning_rate


#     def predict(inputs, weights):
#         summation = np.dot(inputs, weights[1:]) + weights[0]
#         activation = 1.0 if (summation > 0.0) else 0.0

#         return activation


#     def train():
#         for epoch in range(epochs):
#             fail_count = 0
#             i = 0

#             for inputs, label in zip(x, targets):
#                 i = i + 1
#                 prediction = predict(inputs, w)

#                 if (np.sum(np.abs(label - prediction)) != 0):
#                     w[1:] += learning_rate * (label - prediction) * inputs.reshape(inputs.shape[0],1)  
#                     w[0] += learning_rate * (label - prediction)
#                     fail_count += 1




#             if (fail_count == 0):
                
#                 break