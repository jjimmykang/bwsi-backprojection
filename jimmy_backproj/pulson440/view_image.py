import pickle
import argparse
import numpy as np
from math import floor
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter
from matplotlib import transforms
import time
from backproj import backproject_vectorize_real
import copy
from align import open_file


def main():
    parser = argparse.ArgumentParser(description = 'Open a pickle file and show it.')
    parser.add_argument('pickle_dir', help='direct')
    args = parser.parse_args()

    image_data = open_file(args.pickle_dir)
    image_fig = plt.figure()
    #image_ax = image_fig.add_subplot(111)
    plt.imshow(image_data)
    plt.colorbar()
    plt.show()



if __name__ == '__main__':
    main()
