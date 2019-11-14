import matplotlib.pyplot as plt
import argparse
import os
from sklearn.utils import shuffle
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
    print(image_data)
    print(len(image_data))
    # quit()
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


def sigmoid_function(x):
    sigmoid_array = []
    for item in x:
        sigmoid_array.append(1.0/(1 + np.exp(-item)))
    # print(len(sigmoid_array))
    sigmoid_array = np.asarray(sigmoid_array)
    # print(sigmoid_array)
    return sigmoid_array


def sigmoid_derivative(x):
    return x * (1.0 - x)


def get_test_batch(test_data, test_labels, batch_size):
    return_image_batch = []
    return_label_batch = []
    # shuffles data
    test_data, test_labels = shuffle(test_data, test_labels)
    test_data = test_data[:batch_size]
    test_labels = test_labels[:batch_size]
    # converts to numpy arrays
    test_data = np.asarray(test_data)
    test_labels = np.asarray(test_labels)

    return test_data, test_labels


def save_metrics(directory, train_accuracy, test_accuracy, loss, epoch):
    if not os.path.exists(directory):
        os.makedirs(directory)
    train_name = directory + "/" + "train.npy"
    test_name = directory + "/" + "test.npy"
    loss_name = directory + "/" + "loss.npy"
    epoch_name = directory + "/" + "epoch.npy"
    # will hold the data
    train_array = []
    test_array = []
    loss_array = []
    epoch_array = []

    # if the files exist we need to add to them
    if os.path.isfile(train_name) and os.path.isfile(test_name) and os.path.isfile(loss_name) and os.path.isfile(epoch_name):
        # loads the data
        train_array = np.load(train_name)
        test_array = np.load(test_name)
        loss_array = np.load(loss_name)
        epoch_array = np.load(epoch_name)

        # adds the data to the array
        train_array = np.append(train_array, train_accuracy)
        test_array = np.append(test_array, test_accuracy)
        loss_array = np.append(loss_array, loss)
        epoch_array = np.append(epoch_array, epoch)

        # saves the data again
        np.save(train_name, train_array)
        np.save(test_name, test_array)
        np.save(loss_name, loss_array)
        np.save(epoch_name, epoch_array)
    else:
        train_array = np.append(train_array, train_accuracy)
        test_array = np.append(test_array, test_accuracy)
        loss_array = np.append(loss_array, loss)
        epoch_array = np.append(epoch_array, epoch)

        train_array = np.asarray(train_array)
        test_array = np.asarray(test_array)
        loss_array = np.asarray(loss_array)
        epoch_array = np.asarray(epoch_array)

        np.save(train_name, train_array)
        np.save(test_name, test_array)
        np.save(loss_name, loss_array)
        np.save(epoch_name, epoch_array)


def save_metrics_autoencoder(directory, loss_train, loss_test, epoch):
    if not os.path.exists(directory):
        os.makedirs(directory)
    loss_train_name = directory + "/" + "loss_train.npy"
    loss_test_name = directory + "/" + "loss_test.npy"
    epoch_name = directory + "/" + "epoch.npy"
    # will hold the data
    loss_train_array = []
    loss_test_array = []
    epoch_array = []

    # if the files exist we need to add to them
    if os.path.isfile(loss_train_name) and os.path.isfile(epoch_name) and os.path.isfile(loss_test_name):
        # loads the data
        loss_train_array = np.load(loss_train_name)
        loss_test_array = np.load(loss_test_name)
        epoch_array = np.load(epoch_name)

        # adds the data to the array
        loss_train_array = np.append(loss_train_array, loss_train)
        loss_test_array = np.append(loss_test_array, loss_test)
        epoch_array = np.append(epoch_array, epoch)

        # saves the data again
        np.save(loss_train_name, loss_train_array)
        np.save(loss_test_name, loss_test_array)
        np.save(epoch_name, epoch_array)
    else:
        loss_train_array = np.append(loss_train_array, loss_train)
        loss_test_array = np.append(loss_test_array, loss_test)
        epoch_array = np.append(epoch_array, epoch)

        loss_train_array = np.asarray(loss_train_array)
        loss_test_array = np.asarray(loss_test_array)
        epoch_array = np.asarray(epoch_array)

        np.save(loss_train_name, loss_train_array)
        np.save(loss_test_name, loss_test_array)
        np.save(epoch_name, epoch_array)


def get_data_by_label(images, labels, desired_number):
    data_to_return = []
    for i in range(len(images)):
        if labels[i] == desired_number:
            data_to_return.append(images[i])
    return data_to_return
