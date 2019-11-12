from layer import *

import math


class Multi_Layer_Perceptron:
    def __init__(self, input_size, output_size, hidden_layers):
        print("MLP")
        print("input size of ", input_size)
        print("output size of ", output_size)
        print(hidden_layers, " hidden_layers")
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        neurons_in_each_layer = []

        # stores a list of all the neurons at each stage
        neurons_in_each_layer.append(input_size)
        # adds all the hidden layers to the list
        for i in range(hidden_layers):
            layer_number = i + 1
            number_of_neurons = input(
                f"How many neurons for hidden layer #{layer_number}: ")
            number_of_neurons = int(number_of_neurons)
            neurons_in_each_layer.append(number_of_neurons)
        # appends the output layer to the list
        neurons_in_each_layer.append(self.output_size)
        self.neurons_in_each_layer = neurons_in_each_layer

        self.network_layers = []
        # makes the layers
        self.make_layers()

    def make_layers(self):
        for i in range(len(self.neurons_in_each_layer)-1):
            # print(self.neurons_in_each_layer[i])

            self.network_layers.append(
                layer(self.neurons_in_each_layer[i], self.neurons_in_each_layer[i+1]))
            self.network_layers.append(
                sigmoid(self.neurons_in_each_layer[i+1]))
        # print(self.network_layers)
        # print(self.network_layers[0].__dict__)

    def feed_forward(self, inputs):
        layer_outputs = []
        for current_layer in self.network_layers:
            layer_outputs.append(current_layer.feed_forward(inputs))
            # grabs the last layer of outputs
            inputs = layer_outputs[-1]

        return layer_outputs

    def make_prediction(self, inputs):
        # print("will implement")
        activations = self.feed_forward(inputs)
        activations = activations[-1]
        # print(len(activations))
        # quit()
        return activations

    def adjust_prediction(self, predictions):
        adjusted_predictions = []
        print(np.size(predictions))
        for predict in predictions:
            print(len(predict))
            quit()
            if predict > 0.75:
                adjusted_predictions.append(1)
            elif predict < 0.25:
                adjusted_predictions.append(0)
            else:
                adjusted_predictions.append(predict)
        return adjusted_predictions

    def calculate_loss(self, predictions, correct_labels):
        correct_array = [0] * 10
        correct_array[correct_labels] = 1
        for i in range(len(predictions)):
            loss = loss + math.pow(correct_array[i] - predictions[i], 2)
        return loss/len(predictions)

    def calculate_loss_gradient(self, loss):
        return_value = loss * (1-loss)
        return return_value

    def train(self, inputs, correct_class):
        # make predictions
        predictions = self.make_prediction(inputs)
        predictions = self.adjust_prediction(predictions)
        # get loss
        loss = calculate_loss(predictions, correct_class)
        print(loss)
        loss_gradient = calculate_loss_gradient(loss)

        for current_layer in self.network_layers:
            loss_gradient = current_layer.feed_backward(
                current_layer.weights, loss_gradient)

    # def calculate_loss(self, predictions, correct_labels):
    #     correc_array = []
    #     for i in range(len(predictions)):

    #     squared_errors = predictions
    # for i in range(len(predictions)):
    #     for j in range(len(predictions[i])):
    #         if correct_labels[i] == j:
    #             correct_value = 1
    #         else:
    #             correct_value = 0
    #         loss = loss + math.pow(predictions[i][j]-correct_value, 2)
    # loss = loss/2
    # return loss

    # def check_predictions(labels):
