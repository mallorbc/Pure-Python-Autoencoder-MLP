from neuron import *
import numpy as np
from utils import *
import math
import time


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
        # print("set weights")
        # self.weights = np.random.normal(loc=0.0,
        #                                 scale=np.sqrt(
        #                                     2/(self.number_of_inputs+self.number_of_outputs)),
        #                                 size=(self.number_of_inputs, self.number_of_outputs))
        self.weights = np.random.randn(
            self.number_of_inputs, self.number_of_outputs)*0.01
        # print(self.weights)
        # quit()

    def set_biases(self):
        self.biases = np.zeros(self.number_of_outputs)

    def forward(self, inputs):
        # return_values = sigmoid_function(
        #     np.dot(inputs, self.weights) + self.biases)
        # return return_values
        return_value = np.matmul(inputs, self.weights) + self.biases
        return_value = sigmoid_function(return_value)
        #print("forward_return_value", np.shape(return_value))
        # time.sleep(1)
        return return_value

    def backward(self, input, grad_output):
        # print(grad_output)
        # time.sleep(1)
        # print(np.shape(input))
        # time.sleep(1)
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
        return_value = 1/(1+np.exp(-inputs))
        return return_value

    def backward(self, inputs, gradient_output):
        g = 1.0 / (1.0 + np.exp(-inputs))
        g = g*(1-g)
        # print(np.shape(g))
        # time.sleep(1)
        # quit()
        return g
        # return g*gradient_output
        # return_value = inputs * (1-inputs)
        # #print("return_value:", return_value)
        # # quit()
        # if math.isnan(return_value[0][0]):
        #     print("inputs:", inputs)
        #     print("grad_output:", gradient_output)

        #     quit()
        # #return_value = gradient_output*return_value
        # # print(return_value)
        # return return_value


class relu:
    def __init__(self):
        # self.number_of_inputs = inputs
        pass

    def forward(self, input):
        """Apply elementwise ReLU to [batch, input_units] matrix"""
        return np.maximum(0, input)

    def backward(self, input, grad_output):
        """Compute gradient of loss w.r.t. ReLU input"""
        relu_grad = input > 0
        return grad_output*relu_grad
