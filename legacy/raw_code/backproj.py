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
# from backproj_c import backproject_vectorize_c

POINTS2_DATA_DIR = 'data/2Points_1way_data.pkl'
POINTS5_DATA_DIR = 'data/5Points_1way_data.pkl'
MANDRILL_PIC_DIR = 'data/Mandrill_1way_data.pkl'

def backproject_vectorize(data, dimensions):
    '''Backprojects
    DEPRECATED
    DEPRECATED
    DEPRECATED
    Arguments:
        data(dict)
            a dictionary of the data
        dimensions(tuple)
            the resolution of the data (x, y, z)

    Returns:
        return_array(array)
            120x120 array that represents the data
    '''
    x_pixels = dimensions[0]
    y_pixels = dimensions[1]
    z_pixels = dimensions[2]

    # Generate needed objects
    data_array = np.asarray(data['scan_data'])
    return_array = np.empty((x_pixels, y_pixels))
    range_bins = np.asarray(data['range_bins'][0])
    encoded_data = np.empty((x_pixels, y_pixels))
    linear = True

    # Generate coordinate system(120x120 array in which every value is a pair of coordinates(x,y))
    cols = np.arange(-3, 3, 6/y_pixels)
    rows = np.arange(3, -3, -6/x_pixels)
    zetas = z_pixels

    position_map = np.empty((len(rows), len(cols), 3), dtype=np.float32)
    position_map[..., 0] = cols
    position_map[..., 1] = rows[:, None]
    position_map[..., 2] = zetas


    # Broadcast the position map onto a 100x120x120x2 array. It's one hundred of the array to individually calculate the distance for.
    # The first index is the number of pulse.
    # The rest of the array is dimension (120x120x2), representing ordered pairs.
    position_map_3d = np.empty((100, x_pixels, y_pixels, 3))
    position_map_3d[:] = position_map

    # This code processes the platform data from the radar
    # Convert platform positions into 2d
    platform_pos = np.asarray(data['platform_pos'])

    '''
    # Get rid of the z coordinates
    platform_pos_2d = []
    for location in platform_pos:
        platform_pos_2d.append([location[0], location[1]])
    platform_pos_2d = np.asarray(platform_pos_2d)
    '''

    # Convert platform_pos_2d into a 100x120x120x2 array to overlay over position map
    # It's basically the identical layer throughout the 100 pulses
    platform_pos_3d = np.empty((100, x_pixels, y_pixels, 3))
    platform_pos_3d[:, :, :, :] = platform_pos[:, None, None, :]

    # With the platform_map_3d, and position_map_3d, we can generate the distance lookup table.
    distance_lookup_table = np.empty((100, x_pixels, y_pixels))
    distance_lookup_table = np.sqrt(np.power((position_map_3d[..., 0] - platform_pos_3d[..., 0]), 2) + \
                                    np.power((position_map_3d[..., 1] - platform_pos_3d[..., 1]), 2) + \
                                    np.power((position_map_3d[..., 2] - platform_pos_3d[..., 2]), 2))


    #Utilize the distance LUT to generate signal map.
    flattened_distance_lookup = np.reshape(distance_lookup_table, (100, -1))
    signal_matrix = np.empty((100, x_pixels * y_pixels), dtype=np.complex128)
    radar_counter = 0


    while radar_counter < 100:
        signal_matrix[radar_counter] = np.interp(flattened_distance_lookup[radar_counter], range_bins, data_array[radar_counter])
        radar_counter += 1

    signal_matrix = np.abs(np.sum(signal_matrix, axis=0))
    signal_matrix = np.reshape(signal_matrix, (x_pixels, y_pixels))

    #strength_array = np.transpose(strength_array)
    #array = np.asarray([[1, 2, 3], [4, 5, 6]])
    #print(speed.speed_func(array))
    return signal_matrix




def backproject(data, interpolate_arg='linear'):
    '''Backprojects the data shown
    DEPRECATED
    DEPRECATED
    DEPRECATED
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

    print('rows shape:', rows.shape)
    print('cols shape:', cols.shape)

    position_map = np.empty((len(rows), len(cols), 2), dtype=np.float64)
    position_map[..., 0] = rows[:, None]
    position_map[..., 1] = cols

    print('position_map shape:', position_map.shape)

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
                #strength = data_array[radar_counter][index]
                total_strength += strength
                radar_counter += 1

            # Add the absolute value of the complex number to the final array
            encoded_data[x][y] = np.abs(total_strength)

            y += 1
        x += 1

    return_array = encoded_data

    return return_array

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
    parser = argparse.ArgumentParser(description='Process SAR Data.')
    parser.add_argument('file_dir', type=str, help='specify file directory')
    parser.add_argument('x_res', type=int, help='x resolution')
    parser.add_argument('y_res', type=int, help='y resolution')
    parser.add_argument('num_scans', type=int, help='number of scans')
    parser.add_argument('-v', '--visualize', action='store_true', default=0, help='toggle visualization mode')
    args = parser.parse_args()

    # Process command line arguments
    # graph flags
    graph = args.visualize

    # Start timer
    print('started timer')
    start_time = time.time()

    # Open the files
    file_data = open_file(args.file_dir)

    file_to_open = backproject_vectorize(file_data, (args.x_res, args.y_res, 0), args.num_scans)
    print("Finished running backproject()")
    print("--- %s seconds ---" % (time.time() - start_time))

    if graph:
        print('graphing!')

        # Plot
        plt.show()
        image_fig = plt.figure()
        image_ax = image_fig.add_subplot(111)
        h_img = image_ax.imshow(file_to_open)
        plt.show()


if __name__ == '__main__':

    main()
