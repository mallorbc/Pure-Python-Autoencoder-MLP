import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd


def plot_data(directory):
    train_name = directory + "/" + "train.npy"
    test_name = directory + "/" + "test.npy"
    loss_name = directory + "/" + "loss.npy"
    epoch_name = directory + "/" + "epoch.npy"

    train_array = np.load(train_name)
    test_array = np.load(test_name)
    loss_array = np.load(loss_name)
    epoch_array = np.load(epoch_name)

    plt.plot(epoch_array, train_array, label="Train")
    plt.plot(epoch_array, test_array, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch: Test and Train")
    plt.legend()
    plt.show()

    plt.plot(epoch_array, train_array, label="Train")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch: Train")
    plt.legend()
    plt.show()

    plt.plot(epoch_array, test_array, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch: Test")
    plt.legend()
    plt.show()

    plt.plot(epoch_array, loss_array, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.show()


def plot_data_autoencoder(directory):
    loss_train_name = directory + "/" + "loss_train.npy"
    loss_test_name = directory + "/" + "loss_test.npy"

    epoch_name = directory + "/" + "epoch.npy"

    loss_train_array = np.load(loss_train_name)
    loss_test_array = np.load(loss_test_name)

    epoch_array = np.load(epoch_name)

    plt.plot(epoch_array, loss_train_array, label="Loss Train")
    plt.plot(epoch_array, loss_test_array, label="Loss Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.show()


def make_confusion_matrix_array(actual, predicted):
    matrix = confusion_matrix(actual, predicted)
    return matrix


def plot_confusion_matrix(matrix, title):
    print("matrix")
    array = matrix

    df_cm = pd.DataFrame(array, index=[i for i in "0123456789"], columns=[
                         i for i in "0123456789"])

    sn.set(font_scale=1.4)  # for label size
    ax = sn.heatmap(df_cm, annot=True, fmt='g',
                    annot_kws={"size": 20})  # font size
    ax.set(xlabel='Predicted', ylabel='Actual')
    plt.title(title)
    plt.show()


def plot_autoencoder(encoded_images=None, real_images=None, labels=None):
    total_images = []
    total_lables = []

    for image in real_images:
        total_images.append(image)

    for image in encoded_images:
        total_images.append(image)

    for i in range(2):
        for label in labels:
            total_lables.append(label)

    h, w = 28, 28        # for raster image
    nrows, ncols = 2, 8  # array of sub-plots
    figsize = [24, 32]     # figure size, inches

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        img = total_images[i]
        axi.imshow(img, cmap="gray")

        # write row/col indices as axes' title for identification
        axi.set_title("Number: "+str(total_lables[i]))

    plt.tight_layout(True)
    plt.show()


def test_plot(image):
    image = np.reshape(image, (28, 28))
    plt.imshow(image, cmap="gray")
    plt.show()


def plot_features(mlp_images, encoder_images):

    nrows, ncols = 4, 5  # array of sub-plots
    figsize = [12, 18]     # figure size, inches

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        img = mlp_images[i]
        axi.imshow(img, cmap="gray")

        # write row/col indices as axes' title for identification
        axi.set_title("Neuron: "+str(i))

    plt.tight_layout(True)
    plt.show()

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        img = encoder_images[i]
        axi.imshow(img, cmap="gray")

        # write row/col indices as axes' title for identification
        axi.set_title("Neuron: "+str(i))

    plt.tight_layout(True)
    plt.show()


def plot_bar_loss(train_loss, test_loss):
    values = []
    labels = []
    values.append(train_loss)
    values.append(test_loss)
    labels.append("Train")
    labels.append("Test")
    fig, ax = plt.subplots()

    rects1 = ax.bar(labels, values, color='r')
    for rect in rects1:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%3f' % float(height),
                ha='center', va='bottom')

    plt.ylabel('Loss')
    plt.title('Loss: Train and Test')

    # plt.show()
    plt.bar(labels, values)
    # plt.bar(test_loss)
    plt.show()
