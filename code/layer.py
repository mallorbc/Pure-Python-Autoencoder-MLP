from neuron import *
import numpy as np


class layer:
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
        self.learning_rate = 0.01

    def set_weights(self):
        # print("set weights")
        self.weights = np.random.normal(loc=0.0,
                                        scale=np.sqrt(
                                            2/(self.number_of_inputs+self.number_of_outputs)),
                                        size=(self.number_of_inputs, self.number_of_outputs))
        #print("test", len(self.weights[0]))

    def set_biases(self):
        self.biases = np.zeros(self.number_of_outputs)
        # print(len(self.biases))
        # print(self.biases)

    def feed_forward(self, inputs):
        return_values = np.dot(inputs, self.weights) + self.biases
        return return_values

    def feed_backward(self, inputs, gradient_output):
        inputs = np.asarray(inputs)
        grad_input = np.dot(gradient_output, self.weights.T)

        # compute gradient w.r.t. weights and biases
        # COME BACK
        grad_weights = np.dot(inputs.T, gradient_output)
        #grad_weights = np.dot(inputs, gradient_output)
        grad_biases = gradient_output*inputs.shape[0]

        print(np.shape(self.learning_rate))
        print(np.shape(grad_weights))
        print(np.shape(self.weights))
        quit()

        # grad_biases = gradient_output.mean(axis=0)*inputs.shape[0]

        #assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape

        # Here we perform a stochastic gradient descent step.
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input


class sigmoid:
    def __init__(self, inputs):
        self.number_of_inputs = inputs

    def feed_forward(self, inputs):
        return_value = 1/(1+np.exp(-inputs))
        return return_value

    def feed_backward(self, inputs, gradient_output):
        return_value = inputs * (1-inputs)
        return gradient_output*return_value
