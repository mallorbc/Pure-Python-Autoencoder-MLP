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
            # print(len(inputs))
        # print(np.size(np.asarray(layer_outputs)))
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
        # print(predictions)
        # print(len(predictions))
        # quit()
        # predictions = predictions[-1]
        # print(predictions)
        # #print("rows: ", np.size(predictions, 0))
        # for item in predictions:
        #     if item > 0.75:
        #         adjusted_predictions.append(1)
        #     elif item < 0.25:
        #         adjusted_predictions.append(0)
        #     else:
        #         adjusted_predictions.append(item)

        # print(item)
        # print("cols: ", np.size(predictions, 1))

        # #predictions = predictions[-1]
        # print(np.size(predictions))
        # for predict in predictions:
        # for item in predict:
        #     if item > 0.75:
        #         adjusted_predictions.append(1)
        #     elif item < 0.25:
        #         adjusted_predictions.append(0)
        #     else:
        #         adjusted_predictions.append(item)
        # print(adjusted_predictions[0])
        # print(adjusted_predictions[-1])
        # print(len(adjusted_predictions))

        for predict in predictions:
            if predict > 0.75:
                adjusted_predictions.append(1)
            elif predict < 0.25:
                adjusted_predictions.append(0)
            else:
                adjusted_predictions.append(predict)
        return adjusted_predictions

    def calculate_loss(self, predictions, correct_labels):
        loss = 0
        correct_array = [0] * 10
        print("correct label: ", correct_labels)
        correct_array[correct_labels] = 1
        for i in range(len(predictions)):
            loss = loss + math.pow(correct_array[i] - predictions[i], 2)
        print("loss: ", loss)
        print(len(predictions))
        print(predictions)
        return loss/len(predictions)

    def calculate_loss_gradient(self, loss):
        return_value = loss * (1-loss)
        return return_value

    def train(self, inputs, correct_class):
        layer_activations = self.feed_forward(inputs)
        print(len(layer_activations[-1]))
        # quit()
        layer_inputs = [inputs] + layer_activations
        # print(test)
        # print(layer_activations)
        # print(np.size(layer_activations))
        # print(len(layer_activations[0]))
        # print((len(layer_activations[1])))
        # print((len(layer_activations[2])))
        # print((len(layer_activations[3])))

        # quit()
        # make predictions
        predictions = self.make_prediction(inputs)
        predictions = self.adjust_prediction(predictions)
        # get loss
        loss = self.calculate_loss(predictions, correct_class)
        print(loss)
        loss_gradient = self.calculate_loss_gradient(loss)

        for i in range(len(self.network_layers))[::-1]:
            loss_gradient = self.network_layers[i].feed_backward(
                layer_inputs[i], loss_gradient)

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
