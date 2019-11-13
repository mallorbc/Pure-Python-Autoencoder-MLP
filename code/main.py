from utils import *
from mlp import *
from tqdm import trange

import math

import time
from layer import *
from networks import *


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
        network = make_network()

        # converts the data to integers
        loaded_label_data = [int(i) for i in loaded_label_data]
        loaded_text_data = np.asarray(loaded_text_data)
        loaded_label_data = np.asarray(loaded_label_data)

        prediction_array = []
        actual_outputs_array = []
        for epoch in range(100000):
            display = True
            # print("data:", np.shape(loaded_text_data))
            # print("labels", np.shape(loaded_label_data))
            # quit()
            for x_batch, y_batch in iterate_minibatches(loaded_text_data, loaded_label_data, batchsize=1024, shuffle=True):
                # print("input shape: ", np.shape(x_batch))
                # quit()
                # print(y_batch)
                # quit()
                loss = train(network, x_batch, y_batch)
                # time.sleep(1)
                # if epoch % 10 == 0:
                #     print("loss: ", loss, "epoch: ", epoch)
                if epoch % 250 == 0:
                    for i in range(len(x_batch)):
                        prediction_array.append(
                            (predict(network, x_batch[i], display)))
                        display = False
                        actual_outputs_array.append(y_batch[i])
                    print("accuracy: ", get_accuracy(
                        prediction_array, actual_outputs_array), " epoch: ", epoch)
                    print("loss:", loss)
                    prediction_array.clear()
                    actual_outputs_array.clear()
                    # output = predict(network, x_batch[0])
                    # print("Predicted: ", output)
                    # print("Actual: ", y_batch[0])

                    time.sleep(1)
                    break

    else:
        raise SyntaxError("Not a valid run mode")
