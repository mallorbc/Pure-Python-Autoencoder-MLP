import numpy as np
import math
import time
from layer import *


def make_network():
    network = []
    network_input_size = input("How many inputs does the network have? ")
    network_output_size = input("How many outputs does the network have? ")
    hidden_layers = input("How many hidden does the network have? ")
    network_input_size = int(network_input_size)
    network_output_size = int(network_output_size)
    hidden_layers = int(hidden_layers)
    total_layers = 2 + hidden_layers
    hidden_layer_sizes = []
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
            network.append(Dense(hidden_layer_sizes[0], network_output_size))
            network.append(sigmoid_layer())
        else:
            network.append(
                Dense(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
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
    return number_correct/total


def grad_loss(logits, reference_answers):
    # finds the errors and passes it to each layer
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1

    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    return_value = (- ones_for_answers + softmax) / logits.shape[0]

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
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def forward(network, X):
    """
    Compute activations of all network layers by applying them sequentially.
    Return a list of activations for each layer.
    Make sure last activation corresponds to network logits.
    """
    activations = []
    input = X
    for i in range(len(network)):
        activations.append(network[i].forward(X))
        X = network[i].forward(X)

    assert len(activations) == len(network)
    return activations


def predict(network, X, display):
    """
    Compute network predictions.
    """
    logits = forward(network, X)[-1]

    logits = adjust_prediction(logits)
    if display:
        print(logits)
        time.sleep(1)
    return logits.argmax(axis=-1)


def train(network, X, y):
    """
    Train your network on a given batch of X and y.
    You first need to run forward to get all layer activations.
    Then you can run layer.backward going from last to first layer.
    After you called backward for all layers, all Dense layers have already made one gradient step.
    """

    # Get the layer activations
    layer_activations = forward(network, X)
    logits = layer_activations[-1]
    logits = adjust_prediction(logits)
    # loss = softmax_crossentropy_with_logits(logits, y)
    loss = calculate_loss(logits, y)

    # loss_grad = grad_softmax_crossentropy_with_logits(logits, y)
    loss_grad = grad_loss(logits, y)

    for i in range(1, len(network)):
        loss_grad = network[len(
            network) - i].backward(layer_activations[len(network) - i - 1], loss_grad)

    return np.mean(loss)
