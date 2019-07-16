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

POINTS2_DATA_DIR = 'data/2Points_1way_data.pkl'
POINTS5_DATA_DIR = 'data/5Points_1way_data.pkl'
MANDRILL_PIC_DIR = 'data/Mandrill_1way_data.pkl'


def backproject(data):
    '''Backprojects
    Arguments:
        data(dict)
            a dictionary of the data
    '''
    data_array = np.asarray(data['scan_data'])
    return_array = np.zeros((120, 120))
    range_bins = np.asarray(data['range_bins'][0])

    # Generate coordinate system(120x120 array in which every value is a pair of coordinates(x,y))
    rows = np.arange(4, -3, -7/120)
    cols = np.arange(-3, 3, 6/120)

    position_map = np.empty((len(rows), len(cols), 2), dtype=np.float64)
    position_map[..., 0] = rows[:, None]
    position_map[..., 1] = cols
    # Convert platform positions into 2d
    platform_pos = np.asarray(data['platform_pos'])

    # Get rid of the z coordinates
    platform_pos_2d = []
    for location in platform_pos:
        platform_pos_2d.append([location[0], location[1]])

    platform_pos_2d = np.asarray(platform_pos_2d)


    x = 0
    for x_pos in return_array:
        y = 0
        for y_pos in x_pos:

            pos = position_map[x][y]
            total_strength = 0
            radar_counter = 0

            for radar_position in platform_pos_2d:
                distance = np.square(pos[1] - radar_position[0])
                distance += np.square(pos[0] - radar_position[1])
                distance = np.sqrt(distance)
                index = (np.abs(range_bins-distance)).argmin()


                strength = data_array[radar_counter][index]
                total_strength += strength
                radar_counter += 1


            return_array[x][y] = np.abs(total_strength)

            y += 1
        x += 1
    return return_array

def open_files():
    '''Takes the paths of the pickle files and opens them
    Returns:
        twopoint_data(dict)
            Two point scat data

        fivepoint_data(dict)
            Five point scat data

        mandrill_pic(dict)
            Picture of mandrill data
    '''
    with open(POINTS2_DATA_DIR, 'rb') as f:
        twopoint_data = pickle.load(f)
    with open(POINTS5_DATA_DIR, 'rb') as f:
        fivepoint_data = pickle.load(f)
    with open(MANDRILL_PIC_DIR, 'rb') as f:
        mandrill_pic = pickle.load(f)

    return twopoint_data, fivepoint_data, mandrill_pic




def main():
    graph = True
    twopoint_data, fivepoint_data, mandrill_pic = open_files()

    twopoint_image = backproject(mandrill_pic)


    if graph:

        # Plot
        #plt.ioff()
        plt.show()

        rti_fig = plt.figure()
        rti_ax = rti_fig.add_subplot(111)
        h_img = rti_ax.imshow(twopoint_image)
        #rti_ax.set_aspect('auto')
        #rti_ax.set_title('Range-Time Intensity')
        #rti_ax.set_ylabel('Range (m) [Range Bin Number]')
        #rti_ax.set_xlabel('Time Elapsed (s) [Pulse Number]')

        #rti_ax.yaxis.set_major_formatter(signal_formatter)
        #rti_ax.xaxis.set_major_formatter(range_formatter)
        #cbar = rti_fig.colorbar(h_img)
        #cbar.ax.set_ylabel('dB')

        # Show plot
        plt.show()



































if __name__ == '__main__':
    main()
