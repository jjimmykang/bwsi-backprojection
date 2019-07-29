

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
    parser = argparse.ArgumentParser(description='preprocess sar data(with csv also)')
    parser.add_argument('radar_path', type=str, help='path to the unpacked pickle file')
    parser.add_argument('mocap_path', type=str, help='path to the processed mocap data')
    args = parser.parse_args()

    radar_path = args.radar_path
    mocap_path = args.mocap_path
    return_data = {}

    radar_data = open_file(radar_path)
    return_data['scan_data'] =  radar_data['scan_data']
    return_data['scan_timestamps'] = radar_data['timestamps']
    return_data['range_bins'] = radar_data['range_bins']


    mocap_data = open_file(mocap_path)
    mocap_array = np.asarray(mocap_data)
    platform_pos_temp = mocap_array[:, 0]
    platform_pos = np.asarray(list(platform_pos_temp[:]))
    motion_timestamps_temp = mocap_array[:, 1]
    motion_timestamps = np.asarray(list(motion_timestamps_temp[:]))
    print('platform_pos:', platform_pos)

    print('array.shape:', mocap_array.shape)
    print('radar_data keys:', list(radar_data.keys()))
    print('mocap_data type:', type(mocap_data))
























if __name__ == '__main__':
    main()
