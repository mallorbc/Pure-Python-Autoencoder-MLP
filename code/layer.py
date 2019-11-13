from neuron import *
import numpy as np
from utils import *
import math
import time
from scipy.special import expit as sigmoid
from scipy.special import softmax


class Layer:
    def __init__(self):
        """Here you can initialize layer parameters (if any) and auxiliary stuff."""
        # A dummy layer does nothing
        self.weights = np.zeros(shape=(input.shape[1], 10))
        bias = np.zeros(shape=(10,))
        pass

    def forward(self, input):
        """
        Takes input data of shape [batch, input_units], returns output data [batch, 10]
        """
        output = np.matmul(input, self.weights) + bias
        return output


class Dense(Layer):
    def __init__(self, number_of_inputs, number_of_outputs):
        print("new layer with ", number_of_inputs,
              " neruons and ", number_of_outputs, " outputs")
        # sets the number of inputs and outputs
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        # sets the weights for the network
        self.set_weights()
        # sets the biases in the layer
        self.set_biases()
        self.learning_rate = 0.1

    def set_weights(self):
        self.weights = np.random.randn(
            self.number_of_inputs, self.number_of_outputs)*0.01

    def set_biases(self):
        self.biases = np.zeros(self.number_of_outputs)

    def forward(self, inputs):
        # return_values = sigmoid_function(
        #     np.dot(inputs, self.weights) + self.biases)
        # return return_values
        return_value = np.matmul(inputs, self.weights) + self.biases
        # return_value = sigmoid_function(return_value)
        #return_value = softmax_function(return_value)

        #print("forward_return_value", np.shape(return_value))
        # time.sleep(1)
        return return_value

    def backward(self, input, grad_output):

        grad_input = np.dot(grad_output, np.transpose(self.weights))

        # compute gradient w.r.t. weights and biases
        grad_weights = np.transpose(np.dot(np.transpose(grad_output), input))
        grad_biases = np.sum(grad_output, axis=0)

        # Here we perform a stochastic gradient descent step.
        # Later on, you can try replacing that with something better.
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases
        return grad_input


class sigmoid_layer(Layer):
    def __init__(self):
        # self.number_of_inputs = inputs
        pass

    def forward(self, inputs):
        # print(len(inputs))
        # quit()
        return_value = sigmoid(inputs)
        return return_value

    def backward(self, inputs, gradient_output):
        fx = sigmoid(inputs)
        return_value = fx * (1-fx)
        return_value = return_value * gradient_output
        return return_value
