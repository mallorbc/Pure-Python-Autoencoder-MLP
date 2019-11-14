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
    plt.plot(epoch_array, loss_test_array, label="Loss Train")
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
                    annot_kws={"size": 14})  # font size
    ax.set(xlabel='Predicted', ylabel='Actual')
    plt.title(title)
    plt.show()
