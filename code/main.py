from utils import *
from tqdm import trange

import math

import time
from layer import *
from networks import *
from datetime import datetime
# import matplotlib.pyplot as plt


from plot import *


if __name__ == '__main__':
    # plot_autoencoder()
    # quit()
    now = datetime.now()
    dt_string = now.strftime("%m-%d-%H-%M-%S")
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
    parser.add_argument(
        "-ld", "--load", help="directory containing weights to load", default=None, type=str)
    parser.add_argument("-td", "--test_data", default=None,
                        help="file that contains the test data", type=str)
    parser.add_argument("-tl", "--test_labels", default=None,
                        help="file that contains the labels for the test data", type=str)
    parser.add_argument("-pd", "--plot_dir", default=None,
                        help="directory that has the plot data", type=str)
    parser.add_argument(
        "-ldt", "--loadtwo", help="allows us to load the weights of the other model", default=None, type=str)
    # parses the arguments
    args = parser.parse_args()
    mode = args.mode
    data_location = args.data
    labels_location = args.labels
    output_dir = args.output_dir
    weights_to_load = args.load
    test_data_location = args.test_data
    test_labels_location = args.test_labels
    plot_dir = args.plot_dir

    second_weights = args.loadtwo

    if plot_dir is not None:
        plot_dir = os.path.realpath(plot_dir)

    output_dir = output_dir + "/" + dt_string
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # gets the full path of the output
    output_dir = os.path.realpath(output_dir)
    # makes the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if mode < 4 or mode == 11:
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
    elif mode == 3 or mode == 11:

        if test_data_location is not None and test_labels_location is not None:
            # gets the absolute path of the data and the labels
            test_labels_location = os.path.realpath(test_labels_location)
            test_data_location = os.path.realpath(test_data_location)
            # loads the data from the text file
            test_data = load_text_image_file(test_data_location)
            # loads the labels from the text files
            test_labels = load_text_label_file(test_labels_location)

        if weights_to_load is not None:
            if mode == 3:
                network = make_network(
                    weights_to_load)
            elif mode == 11:
                network = []
                full_path_weights = []
                weights = os.listdir(weights_to_load)
                for weight in weights:
                    temp_path = weights_to_load + weight
                    full_path_weights.append(os.path.realpath(temp_path))
                weights = full_path_weights
                network.append(
                    Dense(784, 200, weights[0], False))
                network.append(sigmoid_layer())
                network.append(
                    Dense(200, 10))
                network.append(sigmoid_layer())
            elif mode == 12:
                network = []
                full_path_weights = []
                weights = os.listdir(weights_to_load)
                for weight in weights:
                    temp_path = weights_to_load + weight
                    full_path_weights.append(os.path.realpath(temp_path))
                weights = full_path_weights
                network.append(
                    Dense(784, 200, weights[0]))
                network.append(sigmoid_layer())
                network.append(
                    Dense(200, 10))
                network.append(sigmoid_layer())

        else:
            network = make_network()

        # converts the data to integers
        loaded_label_data = [int(i) for i in loaded_label_data]
        test_labels = [int(i) for i in test_labels]
        # converts to numpy arrays
        loaded_text_data = np.asarray(loaded_text_data)
        loaded_label_data = np.asarray(loaded_label_data)
        test_data = np.asarray(test_data)
        test_labels = np.asarray(test_labels)

        prediction_array = []
        actual_outputs_array = []
        check_point = "checkpoints/checkpoint"
        graph_dir = "graphs"
        output_graph_dir = output_dir + "/" + graph_dir
        if not os.path.exists(output_graph_dir):
            os.makedirs(output_graph_dir)
        # for epoch in range(100000):
        epoch = 0
        print("\n")
        while True:
            display = True
            for x_batch, y_batch in iterate_minibatches(loaded_text_data, loaded_label_data, batchsize=1024, shuffle=True):
                loss = train(network, x_batch, y_batch)

            if epoch % 10 == 0:
                check_point_dir = check_point + str(epoch)
                new_output_dir = output_dir + "/" + check_point_dir

            # gets training batch
            train_data_batch, train_label_batch = get_test_batch(
                loaded_text_data, loaded_label_data, 1024)

            # builds the batch
            for i in range(len(train_data_batch)):
                prediction_array.append(
                    (predict(network, train_data_batch[i], display)))
                display = False
                actual_outputs_array.append(train_label_batch[i])

            display = True
            train_accuracy = get_accuracy(
                prediction_array, actual_outputs_array)

            # clears the array to test on test
            prediction_array.clear()
            actual_outputs_array.clear()

            # tests the test data
            test_data_batch, test_label_batch = get_test_batch(
                test_data, test_labels, 1000)

            # builds the test batch
            for i in range(len(test_data_batch)):
                prediction_array.append(
                    predict(network, test_data_batch[i], display))
                display = False
                actual_outputs_array.append(test_label_batch[i])

            display = True
            # gets test accuracy
            test_accuracy = get_accuracy(
                prediction_array, actual_outputs_array)

            # print("test acc: ", test_accuracy)
            # quit()
            if epoch % 100 == 0 and display:
                print("epoch: ", epoch)
                print("train accuracy: ", train_accuracy)
                print("test accuracy: ", test_accuracy)
                print("loss: ", loss)
                print("\n")
                save_metrics(output_graph_dir, train_accuracy,
                             test_accuracy, loss, epoch)
                display = False
            if epoch % 100 == 0:
                save_weights(network, new_output_dir)
            prediction_array.clear()
            actual_outputs_array.clear()
            # output = predict(network, x_batch[0])
            # print("Predicted: ", output)
            # print("Actual: ", y_batch[0])

            # time.sleep(1)
            # break
            epoch = epoch + 1
    elif mode == 4:
        plot_data(plot_dir)

    elif mode == 5:
        if weights_to_load is None:
            raise SyntaxError("must have weights to load into the model")
        if data_location is None:
            raise SyntaxError("must have training data location")
        if labels_location is None:
            raise SyntaxError("must have training labels location")
        if test_data_location is None:
            raise SyntaxError("must have test data location")
        if test_labels_location is None:
            raise SyntaxError("must have test labels location")

        # gets the absolute path of the data and the labels
        labels_location = os.path.realpath(labels_location)
        data_location = os.path.realpath(data_location)
        # loads the data from the text file
        loaded_text_data = load_text_image_file(data_location)
        # loads the labels from the text files
        loaded_label_data = load_text_label_file(labels_location)

        test_labels_location = os.path.realpath(test_labels_location)
        test_data_location = os.path.realpath(test_data_location)
        # loads the data from the text file
        test_data = load_text_image_file(test_data_location)
        # loads the labels from the text files
        test_labels = load_text_label_file(test_labels_location)

        loaded_label_data = [int(i) for i in loaded_label_data]
        test_labels = [int(i) for i in test_labels]

        train_data = loaded_text_data
        train_labels = loaded_label_data
        train_data = np.asarray(train_data)
        train_labels = np.asarray(train_labels)
        test_data = np.asarray(test_data)
        test_labels = np.asarray(test_labels)

        print("Train: ", np.shape(train_data))
        print("Test: ", np.shape(test_data))

        # all data is now loaded and ready to go
        # now we make a model
        prediction_array = []
        actual_outputs_array = []
        network = make_network(weights_to_load)
        for i in range(len(train_data)):
            prediction_array.append(predict(network, train_data[i], False))
            actual_outputs_array.append(train_labels[i])
        # confusion_matrix for the training data
        train_confusion_matrix = make_confusion_matrix_array(
            actual_outputs_array, prediction_array)

        # make a confusion matrix  for the test data

        prediction_array.clear()
        actual_outputs_array.clear()

        for i in range(len(test_data)):
            prediction_array.append(predict(network, test_data[i], False))
            actual_outputs_array.append(test_labels[i])
        test_confusion_matrix = make_confusion_matrix_array(
            actual_outputs_array, prediction_array)
        # print("train matrix:")
        # print(train_confusion_matrix)
        # print("\n")
        # print("test matrix:")
        # print(test_confusion_matrix)
        plot_confusion_matrix(train_confusion_matrix,
                              "Train confusion matrix")
        plot_confusion_matrix(test_confusion_matrix, "Test confusion matrix")

    elif mode == 6:

        # gets the absolute path of the data and the labels
        labels_location = os.path.realpath(labels_location)
        data_location = os.path.realpath(data_location)
        # loads the data from the text file
        loaded_text_data = load_text_image_file(data_location)
        # loads the labels from the text files
        loaded_label_data = load_text_label_file(labels_location)

        test_labels_location = os.path.realpath(test_labels_location)
        test_data_location = os.path.realpath(test_data_location)
        # loads the data from the text file
        test_data = load_text_image_file(test_data_location)
        # loads the labels from the text files
        test_labels = load_text_label_file(test_labels_location)

        loaded_label_data = [int(i) for i in loaded_label_data]
        test_labels = [int(i) for i in test_labels]

        train_data = loaded_text_data
        train_labels = loaded_label_data
        train_data = np.asarray(train_data)
        train_labels = np.asarray(train_labels)
        test_data = np.asarray(test_data)
        test_labels = np.asarray(test_labels)

        if weights_to_load is not None:
            # print("wrong")
            network = make_network(weights_to_load)
        else:
            network = make_network()

        prediction_array = []
        actual_outputs_array = []
        check_point = "checkpoints/checkpoint"
        graph_dir = "graphs"
        output_graph_dir = output_dir + "/" + graph_dir
        if not os.path.exists(output_graph_dir):
            os.makedirs(output_graph_dir)
        # for epoch in range(100000):
        epoch = 0
        print("\n")
        while True:
            display = True
            for x_batch, y_batch in iterate_minibatches(train_data, train_data, batchsize=1024, shuffle=True):
                loss_train = train_autoencoder(network, x_batch, y_batch)
            if epoch % 100 == 0:
                loss_test = get_loss(network, test_data, test_data)
                print("loss train: ", loss_train)
                print("loss test: ", loss_test)
                print("epoch: ", epoch)
                print("\n")
                check_point_dir = check_point + str(epoch)
                new_output_dir = output_dir + "/" + check_point_dir
                save_weights(network, new_output_dir)
                save_metrics_autoencoder(
                    output_graph_dir, loss_train, loss_test, epoch)

            epoch = epoch + 1

    elif mode == 7:
        plot_data_autoencoder(plot_dir)

    elif mode == 8:
        prediction_array = []
        if weights_to_load is None:
            raise SyntaxError("must have weights to load into the model")
        if test_data_location is None:
            raise SyntaxError("must have test data location")
        if test_labels_location is None:
            raise SyntaxError("must have test labels location")
        # gets the test data

        test_labels_location = os.path.realpath(test_labels_location)
        test_data_location = os.path.realpath(test_data_location)
        test_data = load_text_image_file(test_data_location)
        test_labels = load_text_label_file(test_labels_location)
        test_labels = [int(i) for i in test_labels]

        # load an autoencoder
        network = make_network(weights_to_load)
        # shuffles the dataset
        test_data, test_labels = shuffle_data(test_data, test_labels)
        # grabs 8 images
        test_data = test_data[:8]
        test_labels = test_labels[:8]
        # predicts the images
        for image in test_data:
            prediction_array.append(predict_autoencoder(network, image))
        prediction_array = np.asarray(prediction_array)

        high_range = 0.8
        low_range = 1 - high_range
        for i in range(len(prediction_array)):
            for j in range(len(prediction_array[i])):
                if prediction_array[i][j] > high_range:
                    prediction_array[i][j] = 1
                if prediction_array[i][j] < low_range:
                    prediction_array[i][j] = 0

        # displays the images
        encoded_images = []
        real_images = []
        images_labels = []
        for i in range(len(prediction_array)):
            encoded_images.append(np.reshape(prediction_array[i], (28, 28)))
            real_images.append(np.reshape(test_data[i], (28, 28)))
            images_labels.append(test_labels[i])
        # plots the 8 images real and fake
        plot_autoencoder(encoded_images, real_images, images_labels)

    elif mode == 9:
        if weights_to_load is None:
            raise SyntaxError("must have weights to load into the model")
        if second_weights is None:
            raise SyntaxError(
                "must have second weights to load into the model")
        # second_weights = os.path.realpath(second_weights)
        mlp_network = make_network(weights_to_load)
        encoder_network = make_network(second_weights)
        get_hidden_features(mlp_network, encoder_network)

    elif mode == 10:
        if weights_to_load is None:
            raise SyntaxError("must have weights to load into the model")
        if data_location is None:
            raise SyntaxError("must have training data location")
        if labels_location is None:
            raise SyntaxError("must have training labels location")
        if test_data_location is None:
            raise SyntaxError("must have test data location")
        if test_labels_location is None:
            raise SyntaxError("must have test labels location")

        # gets the absolute path of the data and the labels
        labels_location = os.path.realpath(labels_location)
        data_location = os.path.realpath(data_location)
        # loads the data from the text file
        loaded_text_data = load_text_image_file(data_location)
        # loads the labels from the text files
        loaded_label_data = load_text_label_file(labels_location)

        test_labels_location = os.path.realpath(test_labels_location)
        test_data_location = os.path.realpath(test_data_location)
        # loads the data from the text file
        test_data = load_text_image_file(test_data_location)
        # loads the labels from the text files
        test_labels = load_text_label_file(test_labels_location)

        loaded_label_data = [int(i) for i in loaded_label_data]
        test_labels = [int(i) for i in test_labels]

        train_data = loaded_text_data
        train_labels = loaded_label_data
        train_data = np.asarray(train_data)
        train_labels = np.asarray(train_labels)
        test_data = np.asarray(test_data)
        test_labels = np.asarray(test_labels)

        network = make_network(weights_to_load)

        train_loss = get_loss(network, train_data, train_data)
        test_loss = get_loss(network, test_data, test_data)

        print("train loss: ", train_loss)
        print("test loss: ", test_loss)

        plot_bar_loss(train_loss, test_loss, "Loss: All Data")

        for i in range(10):
            train_batch = get_data_by_label(train_data, train_labels, i)
            test_batch = get_data_by_label(test_data, test_labels, i)
            train_loss = get_loss(network, train_batch, train_batch)
            test_loss = get_loss(network, test_batch, test_batch)
            plot_bar_loss(train_loss, test_loss, "Loss: Number "+str(i))

    else:
        raise SyntaxError("Not a valid run mode")
