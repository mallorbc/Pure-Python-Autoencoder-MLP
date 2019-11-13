from utils import *
from mlp import *
from tqdm import trange

import math

import time
from layer import *


def grad_loss(logits, reference_answers):

    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1
    #print("reference answers: ", reference_answers)
    # quit()
    #print("logits: ", logits)
    #print("ones_for_answers: ", ones_for_answers)
    # quit()

    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    #print("ones_for_answers: ", ones_for_answers)
    # print(logits.shape[0])
    # print(softmax)
    # quit()
    return_value = (- ones_for_answers + softmax) / logits.shape[0]

    # ones_for_answers = np.zeros_like(logits)
    # ones_for_answers[np.arange(len(logits)), reference_answers] = 1

    # g = 1.0 / (1.0 + np.exp(-logits))
    # sigmoid = g*(1-g)
    # return_value = (-ones_for_answers + sigmoid) / logits.shape[0]

    # #print("grad: ", np.shape(return_value))
    # # quit()
    # return return_value
    # print(np.shape(loss))
    # quit()
    # gradient = np.dot(logits.T, loss)/len(loss)
    # print("grad: ", np.shape(gradient))
    # quit()
    # return gradient*-1
    # print(gradient)
    # quit()
    # grad_array = []
    return return_value


def calculate_loss(predictions, correct_labels):
    loss_array = []
    loss = 0
    correct_array = [0] * 10
    correct_array[correct_labels[0]] = 1
    # print("predictions shape:", np.shape(predictions))
    # print("correct labels shape:", np.shape(correct_labels))
    # print(predictions[0])
    # print(correct_labels[0])
    # print(correct_array)
    # quit()

    for i in range(len(correct_labels)):
        correct_array = [0] * 10
        correct_array[correct_labels[i]] = 1
        for j in range(len(predictions[i])):
            #loss = 0
            loss = loss + math.pow(predictions[i][j] - correct_array[j], 2)
            # print(j)
        loss_array.append(loss)
        loss = 0

        correct_array.clear()

    loss_array = np.asarray(loss_array)
    # print(loss_array)
    # print(np.shape(loss_array))
    # quit()
    #print("loss:", np.shape(loss_array))
    # quit()
    return loss_array

    # print("correct label: ", correct_labels)
    # correct_array[correct_labels] = 1
    # for i in range(len(predictions)):
    #     loss = loss + math.pow(correct_array[i] - predictions[i], 2)
    # print("loss: ", loss)
    # # print(len(predictions))
    # # print(predictions)
    # # quit()
    # return loss/len(predictions)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def softmax_crossentropy_with_logits(logits, reference_answers):
    """Compute crossentropy from logits[batch,n_classes] and ids of correct answers"""
    logits_for_answers = logits[np.arange(len(logits)), reference_answers]

    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))

    return xentropy


def grad_softmax_crossentropy_with_logits(logits, reference_answers):
    """Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers"""
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1

    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    return (- ones_for_answers + softmax) / logits.shape[0]


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


def predict(network, X):
    """
    Compute network predictions.
    """
    logits = forward(network, X)[-1]
    return logits.argmax(axis=-1)


def train(network, X, y):
    """
    Train your network on a given batch of X and y.
    You first need to run forward to get all layer activations.
    Then you can run layer.backward going from last to first layer.
    After you called backward for all layers, all Dense layers have already made one gradient step.
    """
    # print("input shape: ", np.shape(X))
    # quit()
    # Get the layer activations
    layer_activations = forward(network, X)
    logits = layer_activations[-1]
    #print("logits shape: ", np.shape(logits))
    # quit()

    # Compute the loss and the initial gradient
    #loss = softmax_crossentropy_with_logits(logits, y)
    loss = calculate_loss(logits, y)
    #print("loss: ", loss)
    #print("loss_shape", np.shape(loss))
    #loss_grad = grad_softmax_crossentropy_with_logits(logits, y)
    loss_grad = grad_loss(logits, y)
    # print(np.shape(loss_grad))
    # quit()
    # print(np.shape(loss_grad))
    # quit()
    #layer_activations = np.asarray(layer_activations)
    # print(np.shape(layer_activations))
    # quit()
    for i in range(1, len(network)):
        # print(layer_activations)
        # quit()
        loss_grad = network[len(
            network) - i].backward(layer_activations[len(network) - i - 1], loss_grad)
        # print(loss_grad)
        # time.sleep(1)
        # quit()

    return np.mean(loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Command line tool for running HW3 for Intelligent Systems")
    parser.add_argument("-m", "--mode", default=1,
                        help="What mode to run the program in", type=int)
    parser.add_argument("-d", "--data", default=None,
                        help="Where the file for the data is; Will be either a txt file or a csv file", type=str)
    parser.add_argument("-l", "--labels", default=None,
                        help="Where the file for the labels is; Will be either a txt file or a csv file", type=str)
    parser.add_argument("-o", "--output_dir", default="../hw_files/output",
                        help="The directory that will contain any generated files", type=str)
    # parses the arguments
    args = parser.parse_args()
    mode = args.mode
    data_location = args.data
    labels_location = args.labels
    output_dir = args.output_dir

    # gets the full path of the output
    output_dir = os.path.realpath(output_dir)
    # makes the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # gets the absolute path of the data and the labels
    labels_location = os.path.realpath(labels_location)
    data_location = os.path.realpath(data_location)
    # loads the data from the text file
    loaded_text_data = load_text_image_file(data_location)
    # loads the labels from the text files
    loaded_label_data = load_text_label_file(labels_location)

    # this displays the images
    if mode == 1:
        # reshapes the data into a 28x28 array
        reshaped_text_data = reshape_image_data(loaded_text_data, False)
        # shuffles the data
        loaded_text_data, loaded_label_data = shuffle_data(
            reshaped_text_data, loaded_label_data)
        # displays the images
        display_images(loaded_text_data, loaded_label_data)

    # this makes the dataset from the original file
    elif mode == 2:
        # reshapes the data into a 28x28 array and flips and rotates it
        reshaped_text_data = reshape_image_data(loaded_text_data, True)
        # shuffles the data
        loaded_text_data, loaded_label_data = shuffle_data(
            reshaped_text_data, loaded_label_data)

        # the training data and labels
        training_text_data = loaded_text_data[:4000]
        training_label_data = loaded_label_data[:4000]
        # the test data and labels
        test_text_data = loaded_text_data[4000:]
        test_label_data = loaded_label_data[4000:]

        # makes the training data_set
        make_train_test_set(
            training_text_data, training_label_data, output_dir, "train")
        # makes the test data set
        make_train_test_set(
            test_text_data, test_label_data, output_dir, "test")

        print("Training and test data set made")

    # this is the mode that creates the MLP
    elif mode == 3:
        # network_input_size = input("How many inputs does the network have? ")
        # network_output_size = input("How many outputs does the network have? ")
        network_input_size = 784
        network_output_size = 10
        hidden_layers = input("How many hidden does the network have? ")
        # makes sure that all the inputs are integers
        network_input_size = int(network_input_size)
        network_output_size = int(network_output_size)
        hidden_layers = int(hidden_layers)

        # network = []
        # network.append(layer(network_input_size, 100))
        # network.append(relu())
        # network.append(layer(100, 10))
        # network.append(relu())

        network = []
        network.append(Dense(network_input_size, 100))
        # network.append(sigmoid_layer())
        network.append(Dense(100, 10))
        # network.append(sigmoid_layer())

        # converts the data to integers
        loaded_label_data = [int(i) for i in loaded_label_data]
        loaded_text_data = np.asarray(loaded_text_data)
        loaded_label_data = np.asarray(loaded_label_data)

        for epoch in range(100000):
            sample = True
            #print("data:", np.shape(loaded_text_data))
            #print("labels", np.shape(loaded_label_data))
            # quit()
            for x_batch, y_batch in iterate_minibatches(loaded_text_data, loaded_label_data, batchsize=64, shuffle=True):
                #print("input shape: ", np.shape(x_batch))
                # quit()
                # print(y_batch)
                # quit()
                loss = train(network, x_batch, y_batch)
                # time.sleep(1)
                if epoch % 10 == 0:
                    print("loss: ", loss, "epoch: ", epoch)
                if epoch != 0 and epoch % 100 == 0 and sample == True:
                    sample = False
                    output = predict(network, x_batch[0])
                    print("Predicted: ", output)
                    print("Actual: ", y_batch[0])
                    time.sleep(4)

        # mlp = Multi_Layer_Perceptron(
        #     network_input_size, network_output_size, hidden_layers)
        # for j in range(100):
        #     for i in range(len(loaded_text_data)):
        #         mlp.train(loaded_text_data[i], int(loaded_label_data[i]))
        #         print("train:", i)
        #     print("loss:", mlp.loss)
        #     time.sleep(1)
    # else this is not a valid mode
    else:
        raise SyntaxError("Not a valid run mode")
