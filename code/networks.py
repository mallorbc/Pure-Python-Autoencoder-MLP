import numpy as np
import math
import time
from layer import *
import os


def make_network(weights_to_load=None):
    network = []
    network_input_size = input("How many inputs does the network have? ")
    network_output_size = input("How many outputs does the network have? ")
    hidden_layers = input("How many hidden does the network have? ")
    network_input_size = int(network_input_size)
    network_output_size = int(network_output_size)
    hidden_layers = int(hidden_layers)
    total_layers = 2 + hidden_layers
    hidden_layer_sizes = []
    if weights_to_load is None:
        # builds the network
        for i in range(hidden_layers):
            layer_number = i + 1
            layer_sizes = input(
                f"how many neurons for hidden layer #{layer_number}?: ")
            layer_sizes = int(layer_sizes)
            hidden_layer_sizes.append(layer_sizes)
        network.append(Dense(network_input_size, hidden_layer_sizes[0]))
        network.append(sigmoid_layer())
        for i in range(len(hidden_layer_sizes)):
            if i == (len(hidden_layer_sizes)-1):
                network.append(
                    Dense(hidden_layer_sizes[0], network_output_size))
                network.append(sigmoid_layer())
            else:
                network.append(
                    Dense(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
                network.append(sigmoid_layer())

    else:
        # gets the full path of the weights
        full_path_weights = []
        weights = os.listdir(weights_to_load)
        for weight in weights:
            temp_path = weights_to_load + weight
            full_path_weights.append(os.path.realpath(temp_path))

        weights = full_path_weights

        # print(weights)
        # quit()
        for i in range(hidden_layers):
            layer_number = i + 1
            layer_sizes = input(
                f"how many neurons for hidden layer #{layer_number}?: ")
            layer_sizes = int(layer_sizes)
            hidden_layer_sizes.append(layer_sizes)
        network.append(
            Dense(network_input_size, hidden_layer_sizes[0], weights[0]))
        network.append(sigmoid_layer())
        for i in range(len(hidden_layer_sizes)):
            if i == (len(hidden_layer_sizes)-1):
                network.append(
                    Dense(hidden_layer_sizes[i], network_output_size, weights[i+1]))
                network.append(sigmoid_layer())
            else:
                network.append(
                    Dense(hidden_layer_sizes[i], hidden_layer_sizes[i+1], weights[i+1]))
                network.append(sigmoid_layer())
    # returns the built network
    return network


def adjust_prediction(predictions):
    return_batch = []
    if predictions.ndim == 2:
        first_dim = predictions.shape[0]
        second_dim = predictions.shape[1]

        for i in range(len(predictions)):
            largest = max(predictions[i])
            smallest = min(predictions[i])
            # print(largest)
            for j in range(len(predictions[i])):
                # predictions[i][j] = predictions[i][j]/largest

                if predictions[i][j] > 0.95:
                    return_batch.append(1)
                elif predictions[i][j] < 0.05:
                    return_batch.append(0)
                # elif predictions[i][j] == largest:
                #     return_batch.append(1)
                else:
                    return_batch.append(predictions[i][j])
        return_batch = np.asarray(return_batch)
        return_batch = return_batch.reshape(first_dim, second_dim)
    else:
        largest = max(predictions)
        smallest = min(predictions)
        if largest < 0.95 and smallest > 0.05:
            for predict in predictions:
                # predict = predict/largest
                if predict > 0.95:
                    return_batch.append(1)
                elif predict < 0.05:
                    return_batch.append(0)
                # elif predict == largest:
                #     return_batch.append(1)
                else:
                    return_batch.append(predict)
        else:
            return_batch = predictions
        return_batch = np.asarray(return_batch)

    return return_batch


def get_accuracy(predictions, actual_results):
    total = len(predictions)
    number_correct = 0
    for i in range(len(predictions)):
        if predictions[i] == actual_results[i]:
            number_correct = number_correct + 1
    return float(number_correct/total)


def grad_loss(outputs, reference_answers):
    # finds the errors and passes it to each layer
    ones_for_answers = np.zeros_like(outputs)
    ones_for_answers[np.arange(len(outputs)), reference_answers] = 1

    error = np.subtract(ones_for_answers, outputs)/outputs.shape[0]

    return_value = error*-1

    # softmax = np.exp(outputs) / np.exp(outputs).sum(axis=-1, keepdims=True)

    # return_value = (- ones_for_answers + softmax) / outputs.shape[0]

    return return_value


def calculate_loss(predictions, correct_labels):
    loss_array = []
    loss = 0
    correct_array = [0] * 10
    correct_array[correct_labels[0]] = 1

    for i in range(len(correct_labels)):
        correct_array = [0] * 10
        correct_array[correct_labels[i]] = 1
        for j in range(len(predictions[i])):
            loss = loss + math.pow(predictions[i][j] - correct_array[j], 2)

        loss_array.append(loss)
        loss = 0

        correct_array.clear()

    loss_array = np.asarray(loss_array)

    return loss_array


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def feed_forward(network, X):
    # calcultes each layers activations and passes them forward
    activations = []
    # passes the inputs through the network
    for i in range(len(network)):
        activations.append(network[i].feed_forward(X))
        X = network[i].feed_forward(X)

    return activations


def predict(network, inputs, display):
    # feeds forward and grabs the last layer
    predictions = feed_forward(network, inputs)[-1]
    # adjusts the predictions
   # predictions = adjust_prediction(predictions)
    # displays the predictions outputs
    # if display:
    #     print(predictions)
    # returns the highest value
    return predictions.argmax(axis=-1)


def train(network, X, y):
    # passes all inputs through the network
    layer_activations = feed_forward(network, X)
    final_outputs = layer_activations[-1]
    # adjust the predictions based on thresholds
    # final_outputs = adjust_prediction(final_outputs)
    # calculates the loss
    loss = calculate_loss(final_outputs, y)

    loss_grad = grad_loss(final_outputs, y)

    # does backpropogation
    for i in range(1, len(network)):
        loss_grad = network[len(
            network) - i].feed_backward(layer_activations[len(network) - i - 1], loss_grad)
    # retunrs the loss
    return np.mean(loss)


def save_weights(network, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    weights = []
    for i in range(len(network)):
        if i % 2 == 0:
            weights.append(network[i].save_weights())

    for i in range(len(weights)):
        file_dir = output_dir + "/" + str(i) + ".npy"
        np.save(file_dir, weights[i])


def train_autoencoder(network, X, y):
    # passes all inputs through the network
    layer_activations = feed_forward(network, X)
    final_outputs = layer_activations[-1]
    # adjust the predictions based on thresholds
    # final_outputs = adjust_prediction(final_outputs)
    # calculates the loss
    loss = calculate_loss_autoencoder(final_outputs, y)

    loss_grad = grad_loss_autoencoder(final_outputs, y)

    # does backpropogation
    for i in range(1, len(network)):
        loss_grad = network[len(
            network) - i].feed_backward(layer_activations[len(network) - i - 1], loss_grad)
    # retunrs the loss
    return np.mean(loss)


def grad_loss_autoencoder(outputs, reference_answers):
    # finds the errors and passes it to each layer
    error = np.subtract(reference_answers, outputs)/outputs.shape[0]

    return_value = error*-1

    return return_value


def calculate_loss_autoencoder(predictions, correct_labels):
    loss_array = []
    loss = 0
    # sums the pixel differences for the batches and returns it
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            loss = loss + math.pow(predictions[i][j] - correct_labels[i][j], 2)
        loss_array.append(loss)
        loss = 0

    loss_array = np.asarray(loss_array)

    return loss_array


def get_loss(network, inputs, labels):
    layer_activations = feed_forward(network, inputs)
    final_outputs = layer_activations[-1]

    loss = calculate_loss_autoencoder(final_outputs, labels)

    loss = np.mean(loss)
    return loss


def predict_autoencoder(network, inputs):
    activations = feed_forward(network, inputs)
    last_layer = activations[-1]
    return last_layer
