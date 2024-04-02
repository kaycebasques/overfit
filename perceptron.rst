.. _perceptron:

=======================================
Journey to the center of the perceptron
=======================================

Building a perceptron from scratch has been the most satisfying experience

This page is my attempt to unify my understanding of perceptrons.
It's a mix between tutorial, explanation, and history.

.. caution::

   Read this as a beginner 

This page is my attempt to understand perceptrons.

.. use excalidraw for diagrams https://excalidraw.com/

----------
Motivation
----------

* `A logical calculus of the ideas immanent in nervous activity <https://www.cs.cmu.edu/~./epxing/Class/10715/reading/McCulloch.and.Pitts.pdf>`_
* `The perceptron: A probabilistic model for information storage and organization in the brain <http://134.208.26.59/INA/A%20probabilistic%20model.pdf>`_
* `Perceptron <https://en.wikipedia.org/wiki/Perceptron>`_
* `Building a Perceptron from Scratch <https://python.plainenglish.io/6b8722807b2e>`_
* `Shallow Neural Network in Keras <https://github.com/the-deep-learners/deep-learning-illustrated/blob/master/notebooks/shallow_net_in_keras.ipynb>`_

.. code-block:: python

   import math
   import random
 
   import numpy as np

 
   class Perceptron:
       def __init__(self, input_size=2, learning_rate=0.01, epochs=5):
           self.learning_rate = learning_rate
           self.epochs = epochs
           self.input_size = input_size
           self.weights = np.random.uniform(-1, 1, size=(input_size))
           self.bias = random.uniform(-1, 1)
 
       def sigmoid(self, X):
           return 1 / (1 + math.exp(-X))

       def z(self, X):
           return np.dot(X, self.weights) + self.bias
 
       def predict(self, X):
           z = self.sigmoid(np.dot(X, self.weights) + self.bias)
           return 1 if z > 0.5 else 0

       def fit(self, X, y):
           for n in range(self.epochs):
               for x, y_actual in zip(X, y):
                   y_predicted = self.predict(x)
                   error = y_actual - y_predicted
                   self.weights += self.learning_rate * error * np.array(x)
                   self.bias += self.learning_rate * error


   X = [[0, 0], [0, 1], [1, 0], [1, 1]]
   y = [0, 1, 1, 1]
   perceptron = Perceptron()
   perceptron.fit(X, y)
