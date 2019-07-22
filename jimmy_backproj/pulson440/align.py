#
# Written by Jimmy Kang
# jimmykang1016@gmail.com
#

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

MANDRILL_1 = 'simulated_misaligned/Mandrill_1way_Misaligned1_data.pkl'
MANDRILL_2 = 'Mandrill_1way_Misaligned2_data.pkl'

def open_file(dir):
    '''Takes the paths of the pickle files and opens them
    Arguments:
        dir(str)
            the path of the file

    Returns:
        return_file(file)
            the file itself
    '''

    with open(dir, 'rb') as f:
        return_file = pickle.load(f)

    return return_file




def main():
    # Argument Parser
    parser = argparse.ArgumentParser(description='align sar data')
    parser.add_argument('--pixel', type=int, help='pixel dimensions each side')
    parser.add_argument('--window', type=float, help='length of real world window')
    parser.add_argument('-v', '--visualize', action='store_true', default=0, help='toggle visualization mode')
    args = parser.parse_args()

    graph = args.visualize

    if args.pixel == None:
        pixel_input = 120
    else:
        pixel_input = args.pixel

    if not args.window == None:
        window_input = int(args.window)
    else:
        window_input = 6

    file_data = open_file(MANDRILL_1)

    #file_opened = backproject_vectorize_real(file_data, pixel_input, simulated=True, window=window_input)


    if graph:
        print('graphing!')

        # Plot
        plt.show()
        image_fig = plt.figure()
        image_ax = image_fig.add_subplot(111)
        h_img = image_ax.imshow(1 * np.log10(np.abs(file_data['scan_data'])), extent=(
            file_data['range_bins'][0 ,0],
            file_data['range_bins'][0, -1],
            file_data['scan_timestamps'][-1]-file_data['scan_timestamps'][0],
            0
        ))

        ranges = np.sqrt(np.sum((file_data['platform_pos'] - file_data['corner_reflector_pos'][0, :]) ** 2, 1))
        plt.plot(ranges, file_data['motion_timestamps'], 'r--')
        plt.show()



if __name__ == '__main__':
    main()
