import matplotlib.pyplot as plt
import argparse
import os
import cv2
import re
import numpy as np


def load_text_image_file(path_to_file):
    all_data = []
    with open(path_to_file) as f:
        all_lines = f.read().splitlines()
    for lines in all_lines:
        # splits the lines based on tab
        lines = lines.split('\t')
        # converts all the values in the list to a float value insted of a string value
        for index, item in enumerate(lines):
            # converts the data
            lines[index] = float(item)
        # adds the converted data to a list
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


def reshape_image_data(image_data):
    reshaped_data = []
    for image in image_data:
        # reshapes the image into a 28x28 2d array
        reshaped_image = np.reshape(image, (28, 28))
        # flips the image
        reshaped_image = np.flipud(reshaped_image)
        # rotates the image 270 degrees
        reshaped_image = np.rot90(reshaped_image, 3)
        # reshaped_image = reshaped_image.flatten()
        reshaped_data.append(reshaped_image)
    print("Image data has been reshaped to 28x28")
    return reshaped_data


def display_images(image_data, image_label):
    print("displaying images")
    for i in range(len(image_data)):
        print("Image Label: ", image_label[i])
        img = image_data[i]
        # img = np.reshape(img, (28, 28))
        plt.imshow(img, cmap=plt.cm.binary)
        plt.show()


def make_train_test_set(image_data, image_label):
    print("Make data")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Command line tool for running HW3 for Intelligent Systems")
    parser.add_argument("-m", "--mode", default=1,
                        help="What mode to run the program in", type=int)
    parser.add_argument("-d", "--data", default=None,
                        help="Where the file for the data is; Will be either a txt file or a csv file", type=str)
    parser.add_argument("-l", "--labels", default=None,
                        help="Where the file for the labels is; Will be either a txt file or a csv file", type=str)
    # parses the arguments
    args = parser.parse_args()
    mode = args.mode
    data_location = args.data
    labels_location = args.labels

    # gets the absolute path of the data and the labels
    labels_location = os.path.realpath(labels_location)
    data_location = os.path.realpath(data_location)
    # loads the data from the text file
    loaded_text_data = load_text_image_file(data_location)
    # loads the labels from the text files
    loaded_label_data = load_text_label_file(labels_location)
    # print(data_location)
    # print(labels_location)
    reshaped_text_data = reshape_image_data(loaded_text_data)
    display_images(reshaped_text_data, loaded_label_data)
    # print(reshaped_text_data)
