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

POINTS2_DATA_DIR = 'data/2Points_1way_data.pkl'
POINTS5_DATA_DIR = 'data/5Points_1way_data.pkl'
MANDRILL_PIC_DIR = 'data/Mandrill_1way_data.pkl'

def backproject_vectorize(data, interpolate_arg):
    '''Backprojects
    Arguments:
        data(dict)
            a dictionary of the data
        interpolate(str)
            'linear' - linear interpolation
        regularize(bool)
            activate regularization down to decimals

    Returns:
        return_array(array)
            120x120 array that represents the data
    '''
    # Generate needed objects
    data_array = np.asarray(data['scan_data'])
    return_array = np.zeros((120, 120))
    range_bins = np.asarray(data['range_bins'][0])
    encoded_data = np.zeros((120, 120))


    position_map_3d = np.empty((120, 120, 100))
    # Generate coordinate system(120x120 array in which every value is a pair of coordinates(x,y))
    rows = np.arange(4, -2, -6/120)
    cols = np.arange(-3, 3, 6/120)

    position_map = np.empty((len(rows), len(cols), 2), dtype=np.float64)
    position_map[..., 0] = rows[:, None]
    position_map[..., 1] = cols


    position_map_3d

    # Convert platform positions into 2d
    platform_pos = np.asarray(data['platform_pos'])

    # Get rid of the z coordinates
    platform_pos_2d = []
    for location in platform_pos:
        platform_pos_2d.append([location[0], location[1]])

    platform_pos_2d = np.asarray(platform_pos_2d)

    # Iterate through every pixel, with counters
    x = 0
    for x_pos in return_array:
        y = 0
        for y_pos in x_pos:
            # Get real coordinates from pixel
            pos = position_map[x][y]
            total_strength = 0
            radar_counter = 0

            # Iterate through every radar position
            for radar_position in platform_pos_2d:
                # Calculate distance from radar to the current pixel position
                distance = np.square(pos[1] - radar_position[0])
                distance += np.square(pos[0] - radar_position[1])
                distance = np.sqrt(distance)
                index = (np.abs(range_bins-distance)).argmin()

                # Check linear interpolation flag, and interpolates if activated
                if linear:
                    strength = np.interp(distance, range_bins, data_array[radar_counter])

                # Find the signal strength and then add it to the total counter
                strength = data_array[radar_counter][index]
                total_strength += strength
                radar_counter += 1

            # Add the absolute value of the complex number to the final array
            encoded_data[x][y] = np.abs(total_strength)

            y += 1
        x += 1

    return_array = encoded_data

    return return_array

def backproject(data, interpolate_arg):
    '''Backprojects
    Arguments:
        data(dict)
            a dictionary of the data
        interpolate(str)
            'linear' - linear interpolation
        regularize(bool)
            activate regularization down to decimals

    Returns:
        return_array(array)
            120x120 array that represents the data
    '''
    # Generate needed objects
    data_array = np.asarray(data['scan_data'])
    return_array = np.zeros((120, 120))
    range_bins = np.asarray(data['range_bins'][0])
    encoded_data = np.zeros((120, 120))

    # Process interpolate arguments
    if interpolate_arg == 'linear':
        linear = True
    else:
        linear = False

    # Generate coordinate system(120x120 array in which every value is a pair of coordinates(x,y))
    rows = np.arange(4, -2, -6/120)
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

    # Iterate through every pixel, with counters
    x = 0
    for x_pos in return_array:
        y = 0
        for y_pos in x_pos:
            # Get real coordinates from pixel
            pos = position_map[x][y]
            total_strength = 0
            radar_counter = 0

            # Iterate through every radar position
            for radar_position in platform_pos_2d:
                # Calculate distance from radar to the current pixel position
                distance = np.square(pos[1] - radar_position[0])
                distance += np.square(pos[0] - radar_position[1])
                distance = np.sqrt(distance)
                index = (np.abs(range_bins-distance)).argmin()

                # Check linear interpolation flag, and interpolates if activated
                if linear:
                    strength = np.interp(distance, range_bins, data_array[radar_counter])

                # Find the signal strength and then add it to the total counter
                strength = data_array[radar_counter][index]
                total_strength += strength
                radar_counter += 1

            # Add the absolute value of the complex number to the final array
            encoded_data[x][y] = np.abs(total_strength)

            y += 1
        x += 1

    return_array = encoded_data

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
    # Start timer
    start_time = time.time()

    # Argument Parser
    parser = argparse.ArgumentParser(description='Process SAR Data.')
    args = parser.parse_args()

    # graph flags
    graph = True

    # Open the files
    twopoint_data, fivepoint_data, mandrill_pic = open_files()
    file_to_open = backproject(twopoint_data, 'linear')
    print(file_to_open)

    print("Finished running backproject()")
    print("--- %s seconds ---" % (time.time() - start_time))

    if graph:

        # Plot
        #plt.ioff()
        plt.show()

        rti_fig = plt.figure()
        rti_ax = rti_fig.add_subplot(111)
        h_img = rti_ax.imshow(file_to_open)
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
