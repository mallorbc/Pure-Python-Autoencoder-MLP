from layer import *

import math
import time


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
            # self.network_layers.append(
            #     sigmoid(self.neurons_in_each_layer[i+1]))
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
        correct_array[correct_labels] = 1
        return_array = []
        diff_array = []

        print("correct label: ", correct_labels)
        correct_array[correct_labels] = 1
        for i in range(len(predictions)):
            loss = loss + math.pow(correct_array[i] - predictions[i], 2)
        print("loss: ", loss)
        # print(len(predictions))
        # print(predictions)
        # quit()
        return loss/len(predictions)
        # for i in range(len(predictions)):
        #     return_array.append(math.pow(predictions[i] - correct_array[i], 2))
        # #return_array = math.pow(predictions - correct_array, 2)
        # return return_array
        # # print(return_array)
        # # quit()

    def calculate_loss_gradient(self, loss):
        return_array = []
        for i in range(len(loss)):
            value = loss[i] * (1-loss[i])
            return_array.append(value)
        # return_value = loss * (1-loss)
        print(return_array)

        # quit()
        return return_array

    def train(self, inputs, correct_class):
        layer_activations = self.feed_forward(inputs)
        print(len(layer_activations[-1]))
        logits = layer_activations[-1]
        # print(logits)
        # loss = self.softmax_crossentropy_with_logits(logits, correct_class)
        # loss_grad = self.grad_softmax_crossentropy_with_logits(
        #  logits, correct_class)

        # quit()
        layer_inputs = [inputs] + layer_activations

        # loss = self.softmax_crossentropy_with_logits(logits, correct_class)
        # loss_grad = grad_softmax_crossentropy_with_logits(
        #     logits, correct_class)

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
        print(predictions)
        # time.sleep(1)
        # get loss
        loss = self.calculate_loss(predictions, correct_class)
        self.loss = loss
        # print(loss)
        loss_gradient = self.calculate_loss_gradient(loss)
        # loss_grad = sigmoid_derivative(loss)

        loss
        # quit()
        print("testing", np.shape(self.network_layers[-1].weights))

        for layer_index in range(len(self.network_layers))[::-1]:
            layer = self.network_layers[layer_index]

            loss_grad = layer.feed_backward(
                layer_inputs[layer_index], loss_grad)

        # for i in range(len(self.network_layers))[::-1]:
        #     loss_gradient = self.network_layers[i].feed_backward(
        #         layer_inputs[i], loss_gradient)

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

    def softmax_crossentropy_with_logits(self, logits, reference_answers):
        # Compute crossentropy from logits[batch,n_classes] and ids of correct answers
        logits_for_answers = logits[np.arange(len(logits)), reference_answers]

        xentropy = - logits_for_answers + \
            np.log(np.sum(np.exp(logits), axis=-1))

        return xentropy

    def grad_softmax_crossentropy_with_logits(self, logits, reference_answers):
        # Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers
        ones_for_answers = np.zeros_like(logits)
        ones_for_answers[np.arange(len(logits)), reference_answers] = 1

        softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

        return (- ones_for_answers + softmax) / logits.shape[0]
