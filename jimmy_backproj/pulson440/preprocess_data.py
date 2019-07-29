

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
    # Parse arguments
    parser = argparse.ArgumentParser(description='preprocess sar data(with csv also)')
    parser.add_argument('folder_path', type=str, help='path to folder')
    parser.add_argument('radar_path', type=str, help='name of unpacked radar pickle file')
    parser.add_argument('mocap_path', type=str, help='name of unpacked data file')
    parser.add_argument('first_reflector_path', type=str, help='path to the first reflector mocap data')
    parser.add_argument('second_reflector_path', type=str, help='path to the second reflector mocap data')
    args = parser.parse_args()

    # Fetch paths
    radar_path = args.folder_path + '/' + args.radar_path
    mocap_path = args.folder_path + '/' + args.mocap_path
    return_data = {}

    # Fetch radar data and put into dictionary
    radar_data = open_file(radar_path)
    return_data['scan_data'] =  np.asarray(radar_data['scan_data'])
    return_data['scan_timestamps'] = np.asarray(radar_data['timestamps'])
    return_data['range_bins'] = np.asarray(radar_data['range_bins'])
    print('scan_data.shape:', np.asarray(radar_data['scan_data']).shape)
    print('scan_timestamps.shape:', np.asarray(radar_data['timestamps']).shape)
    print('range_bins.shape:', np.asarray(radar_data['range_bins']).shape)

    # Fetch mocap data(platform positions and timestamps)
    mocap_data = open_file(mocap_path)
    mocap_array = np.asarray(mocap_data)
    platform_pos_temp = mocap_array[:, 0]
    platform_pos = np.asarray(list(platform_pos_temp[:]))
    print('platform_pos.shape:', platform_pos.shape)
    motion_timestamps_temp = mocap_array[:, 1]
    motion_timestamps = np.asarray(list(motion_timestamps_temp[:]))
    print('motion_timestamps.shape:', motion_timestamps.shape)
    return_data['platform_pos'] = platform_pos
    return_data['motion_timestamps'] = motion_timestamps

    # Fetch platform data
    corner_reflector_pos = np.empty((2, 3))
    reflector_1 = open_file(args.folder_path + '/' + args.first_reflector_path)
    reflector_1_mocap = np.asarray(reflector_1)
    corner_reflector_pos[0, :] = np.asarray(reflector_1[0][0])

    reflector_2 = open_file(args.folder_path + '/' + args.second_reflector_path)
    reflector_2_mocap = np.asarray(reflector_2)
    corner_reflector_pos[1, :] = np.asarray(reflector_2[0][0])

    return_data['corner_reflector_pos'] = corner_reflector_pos

    # Paste output to pickle file in same directory
    MASTER_PATH = args.folder_path + '/master_data.pkl'
    with open(MASTER_PATH, 'wb') as p:
        pickle.dump(return_data, p)


if __name__ == '__main__':
    main()
