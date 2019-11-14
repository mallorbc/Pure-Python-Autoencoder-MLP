from utils import *
from tqdm import trange

import math

import time
from layer import *
from networks import *
from datetime import datetime


if __name__ == '__main__':
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
    # parses the arguments
    args = parser.parse_args()
    mode = args.mode
    data_location = args.data
    labels_location = args.labels
    output_dir = args.output_dir
    weights_to_load = args.load
    test_data_location = args.test_data
    test_labels_location = args.test_labels

    output_dir = output_dir + "/" + dt_string
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

        if test_data_location is not None and test_labels_location is not None:
            # gets the absolute path of the data and the labels
            test_labels_location = os.path.realpath(labels_location)
            test_data_location = os.path.realpath(data_location)
            # loads the data from the text file
            test_data = load_text_image_file(test_data_location)
            # loads the labels from the text files
            test_labels = load_text_label_file(test_labels_location)

        if weights_to_load is not None:
            # print("wrong")
            network = make_network(weights_to_load)
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

            #print("test acc: ", test_accuracy)
            # quit()
            if epoch % 100 == 0 and display:
                print("epoch: ", epoch)
                print("train accuracy: ", train_accuracy)
                print("test accuracy: ", test_accuracy)
                print("loss: ", loss)
                print("\n")
                display = False

            save_weights(network, new_output_dir)
            prediction_array.clear()
            actual_outputs_array.clear()
            # output = predict(network, x_batch[0])
            # print("Predicted: ", output)
            # print("Actual: ", y_batch[0])

            # time.sleep(1)
            # break
            epoch = epoch + 1

    else:
        raise SyntaxError("Not a valid run mode")
