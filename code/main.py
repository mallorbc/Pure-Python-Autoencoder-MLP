import matplotlib.pyplot as plt
import argparse
import os
import cv2
import re

def load_text_file(path_to_file):
    all_data = []
    with open(path_to_file) as f:
        all_lines = f.read().splitlines()
    for lines in all_lines:
        #splits the lines based on tab
        lines = lines.split('\t')
        #converts all the values in the list to a float value insted of a string value
        for index, item in enumerate(lines):
            #converts the data
            lines[index] = float(item)
        #adds the converted data to a list
        all_data.append(lines)
    print("All data loaded and converted to float")

    # print(len(all_data))
    # print(len(all_data[0]))





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Command line tool for running HW3 for Intelligent Systems")
    parser.add_argument("-m","--mode",default=1, help="What mode to run the program in",type=int)
    parser.add_argument("-d","--data",default=None, help="Where the file for the data is; Will be either a txt file or a csv file",type=str)
    parser.add_argument("-l","--labels",default=None, help="Where the file for the labels is; Will be either a txt file or a csv file",type=str)
    args = parser.parse_args()
    mode = args.mode
    data_location = args.data
    labels_location = args.labels

    #gets the absolute path of the data and the labels
    data_location = os.path.abspath(data_location)
    labels_location = os.path.abspath(labels_location)
    load_text_file(data_location)
    print(data_location)
    print(labels_location)