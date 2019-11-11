import matplotlib.pyplot as plt
import argparse
import os
import cv2
from sklearn.utils import shuffle
import numpy as np
import time


def load_text_image_file(path_to_file):
    all_data = []
    with open(path_to_file) as f:
        all_lines = f.read().splitlines()
    for lines in all_lines:
        # splits the lines based on tab
        lines = lines.split('\t')
        # converts all the values in the list to a float value insted of a string value
        for index, item in enumerate(lines):
            if index == 784:
                break
            lines[index] = float(item)
        # adds the converted data to a list
        # for loading a custom data file the size is larger
        if len(lines) == 785:
            lines = lines[:len(lines)-1]
        all_data.append(lines)
    print("All data loaded and converted to float")
    return all_data


def load_text_label_file(path_to_file):
    all_data = []
    print(path_to_file)
    # quit()
    with open(path_to_file) as f:
        all_lines = f.read().splitlines()
    return all_lines


def reshape_image_data(image_data, flip_and_rotate=False):
    reshaped_data = []
    for image in image_data:
        # reshapes the image into a 28x28 2d array
        reshaped_image = np.reshape(image, (28, 28))
        if flip_and_rotate:
            # flips the image
            reshaped_image = np.flipud(reshaped_image)
            # rotates the image 270 degrees
            reshaped_image = np.rot90(reshaped_image, 3)
        # reshaped_image = reshaped_image.flatten()
        reshaped_data.append(reshaped_image)
    # print("Image data has been reshaped to 28x28")
    return reshaped_data


def display_images(image_data, image_label):
    print("displaying images")
    for i in range(len(image_data)):
        print("Image Label: ", image_label[i])
        img = image_data[i]
        # img = np.reshape(img, (28, 28))
        plt.imshow(img, cmap=plt.cm.binary)
        plt.show()


def shuffle_data(image_data, image_label):
    # shuffles the data together
    shuffled_image_data, shuffled_image_label = shuffle(
        image_data, image_label)
    return shuffled_image_data, shuffled_image_label


def make_train_test_set(image_data, image_label, output_dir, file_name):
    confirm = 1
    final_file_name = file_name + "_images.txt"
    # checks to see if the file already exists to make sure we don't overwrite it
    if os.path.isfile(output_dir + "/" + final_file_name):
        confirm = 0
        print(final_file_name, " already exists, you you want to replace it?")
        confirm = confirm_or_deny()
    if confirm == 1:
        # makes the new dataset
        make_image_dataset(image_data, output_dir, final_file_name)

    final_file_name = file_name + "_labels.txt"
    # checks to see if the file already exists to make sure we don't overwrite it
    if os.path.isfile(output_dir + "/" + final_file_name):
        confirm = 0
        print(final_file_name, " already exists, you you want to replace it?")
        confirm = confirm_or_deny()
    if confirm == 1:
        # makes the new dataset
        make_labels_dataset(image_label, output_dir, final_file_name)


def make_image_dataset(image_data, output_dir, file_name):
    # creates a text file of the image data
    output_file = output_dir + "/" + file_name
    with open(output_file, "w") as f:
        for image in image_data:
            # flattens the array to a 1D array
            image = image.flatten()
            for number in image:
                f.write(str(number) + "\t")
            f.write("\n")


def make_labels_dataset(label_data, output_dir, file_name):
    # creates a text file of the labels
    output_file = output_dir + "/" + file_name
    with open(output_file, "w") as f:
        for label in label_data:
            f.write(str(label))
            f.write("\n")


def confirm_or_deny():
    value = input("Please confirm.(1/0)")
    value = int(value)
    if value != 0 and value != 1:
        print("Either 0 or 1")
        value = confirm_or_deny()
    value = int(value)
    return value


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
