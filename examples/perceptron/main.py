from math import e
import random

import numpy as np


class Perceptron:

    def __init__(self, input_size=2, learning_rate=0.01, epochs=5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.input_size = input_size
        self.weights = np.random.uniform(-1, 1, size=(input_size))
        self.bias = random.uniform(-1, 1)

    def z(self, X):
        return np.dot(X, self.weights) + self.bias

    def sigmoid(self, z):
        return 1 / (1 + (e**-z))

    def predict(self, X):
        activation = self.sigmoid(self.z(X))
        return 1 if activation > 0.5 else 0

    def fit(self, X, y):
        for n in range(self.epochs):
            print(f'=======')
            print(f'epoch {n+1}')
            print(f'=======')
            for x, y_actual in zip(X, y):
                y_predicted = self.predict(x)
                print(f'x: {x}, y_actual: {y_actual}, y_predicted: {y_predicted}')
                error = y_actual - y_predicted
                print(f'error: {error}')
                print(f'weights before: {self.weights}')
                self.weights += self.learning_rate * error * np.array(x)
                print(f'weights after:  {self.weights}')
                print(f'bias before: {self.bias}')
                self.bias += self.learning_rate * error
                print(f'bias after:  {self.bias}')
                print(f'--------------------------------------')
            print()


# Actually interesting data would be IEF, GLD (X) and VTI (y)
# Would it be OK to add more inputs? UUP, USO, VNQ
# Convert to 1 if positive change from day before else 0
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 1]
perceptron = Perceptron(epochs=100)
perceptron.fit(X, y)
