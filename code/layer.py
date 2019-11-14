import numpy as np
from utils import *
import math
import time
from scipy.special import expit as sigmoid
from scipy.special import softmax


class Layer:
    def __init__(self):
        # does nothing
        pass


class Dense(Layer):
    def __init__(self, number_of_inputs, number_of_outputs, weights_to_load=None):
        print("new layer with ", number_of_inputs,
              " neruons and ", number_of_outputs, " outputs")
        # sets the number of inputs and outputs
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        # sets the weights for the network
        if weights_to_load is None:
            self.set_weights()
            # print(self.weights)

        else:
            self.load_weights(weights_to_load)
            # print(self.weights)

        # sets the biases in the layer
        self.set_biases()
        self.learning_rate = 0.1
        self.momentum = 0.5
        self.previous_weight_change = 0

    def set_weights(self):
        self.weights = np.random.randn(
            self.number_of_inputs, self.number_of_outputs)*0.01

    def set_biases(self):
        self.biases = np.zeros(self.number_of_outputs)

    def feed_forward(self, inputs):
        return_value = np.matmul(inputs, self.weights) + self.biases

        return return_value

    def feed_backward(self, input, grad_output):

        grad_input = np.dot(grad_output, np.transpose(self.weights))

        grad_weights = np.transpose(np.dot(np.transpose(grad_output), input))
        grad_biases = np.sum(grad_output, axis=0)

        # gradient descent step.
        weight_change = (self.momentum * self.previous_weight_change) + \
            (self.learning_rate * grad_weights)
        self.weights = self.weights - weight_change
        # updates old weight change for momentum
        self.previous_weight_change = weight_change
        self.biases = self.biases - self.learning_rate * grad_biases
        return grad_input

    def save_weights(self):
        return self.weights

    def load_weights(self, weights_to_load):
        self.weights = np.load(weights_to_load)
        print("Loaded weights")

    def get_weights(self):
        return self.weights


class sigmoid_layer(Layer):
    def __init__(self):
        # self.number_of_inputs = inputs
        print("Sigmoid layer added")

    def feed_forward(self, inputs):
        # print(len(inputs))
        # quit()
        return_value = sigmoid(inputs)
        return return_value

    def feed_backward(self, inputs, gradient_output):
        fx = sigmoid(inputs)
        return_value = fx * (1-fx)
        return_value = return_value * gradient_output
        return return_value
