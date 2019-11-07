import matplotlib.pyplot as plt
import argparse
import os
import cv2
import re

def load_text_file(path_to_file):
    all_data = []
    float_data = []
    print("loaded")
    with open(path_to_file) as f:
        all_lines = f.read().splitlines()
    for lines in all_lines:
        lines = lines.split('\t')
        # map(float, lines)
        # # [float(i) for i in lines]
        # # for entry in lines:
        # #     lines = float(entry)
        # print(type(lines[0]))
        # print(len(lines))
        # quit()

    # file_to_read = open(path_to_file,"r")
    # all_file_lines = file_to_read.readlines()
    # for line in all_file_lines:
    #     re.split(r"\t+",line)
    #     print(len(line))
    #     print(line)
    #     quit()
    #     # split_line = line.split(" ")
    #     # print(len(split_line))
    # quit()




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