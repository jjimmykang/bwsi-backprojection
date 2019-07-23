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

def compute_ranges(file_data):
    # Compute the range trajectories of both corner reflectors to plot
    ranges_1 = np.sqrt(np.sum(
            (file_data['platform_pos'] - file_data['corner_reflector_pos'][0, :])**2, 1))
    ranges_2 = np.sqrt(np.sum(
            (file_data['platform_pos'] - file_data['corner_reflector_pos'][1, :])**2, 1))

    return ranges_1, ranges_2

def visualize_data(file_data):

    scan_data = file_data['scan_data']
    motion_timestamps = file_data['motion_timestamps']


    print('graphing!')

    # Plot
    plt.show()
    image_fig = plt.figure()
    image_ax = image_fig.add_subplot(111)
    h_img = image_ax.imshow(1 * np.log10(np.abs(scan_data)), extent=(
        file_data['range_bins'][0 ,0],
        file_data['range_bins'][0, -1],
        file_data['scan_timestamps'][-1]-file_data['scan_timestamps'][0],
        0
    ),
    zorder=5)

    ranges_1, ranges_2 = compute_ranges(file_data)

    # Plot the range trajectory of corner reflector 1
    plt.plot(ranges_1, # x-values
             motion_timestamps, # y-values
             'r--', # Line format
             label='Corner Reflector 1',
             zorder=20) # Legend label

    # Plot the range trajectory of corner reflector 2
    plt.plot(ranges_2, # x-values
             motion_timestamps, # y-values
             'g--', # Line format
             label='Corner Reflector 2',
             zorder=20) # Legend label

    plt.xlabel('Range (m)')
    plt.ylabel('Elapsed Time (s)')
    plt.title('Alignment View')
    # Add a legend
    plt.legend()
    plt.show()

def find_nearest(array, value):

    idx = (np.abs(array - value)).argmin()
    return idx

def main():
    # Argument Parser
    parser = argparse.ArgumentParser(description='align sar data')
    parser.add_argument('--pixel', type=int, help='pixel dimensions each side')
    parser.add_argument('--window', type=float, help='length of real world window')
    parser.add_argument('-v', '--visualize', action='store_true', default=0, help='toggle visualization mode')
    parser.add_argument('--shift', type=float, help='amount to shift the data by')
    args = parser.parse_args()

    # Command line argument processing
    graph = args.visualize

    if args.pixel == None:
        pixel_input = 120
    else:
        pixel_input = args.pixel

    if not args.window == None:
        window_input = int(args.window)
    else:
        window_input = 6

    if not args.shift == None:
        align_amt = args.shift
    else:
        align_amt = 0


    file_data = open_file(MANDRILL_1)


    # Subtract least term from timestamps(to regularize at 0)
    scan_timestamps = file_data['scan_timestamps'] - file_data['scan_timestamps'][0]
    motion_timestamps = file_data['motion_timestamps'] - file_data['motion_timestamps'][0]

    # Import data as variables
    platform_pos = file_data['platform_pos']
    scan_data = file_data['scan_data']
    # Set shape
    scan_data_shape =  file_data['scan_data'].shape
    # Get the number of scans
    num_scans = scan_data_shape[0]

    # Shift the data manually
    # Fetch align amount required
    shift_time = align_amt

    # Translate the shifted time to indexes for each of the timestamp lists
    shift_motion_index = find_nearest(motion_timestamps, shift_time)
    shift_scan_index = find_nearest(scan_timestamps, shift_time)

    # Slice the data arrays
    # Motion/platform position data by the index
    motion_timestamps = motion_timestamps[shift_motion_index:]
    platform_pos = platform_pos[shift_motion_index:]

    # Re-regularize(because it got sliced)
    motion_timestamps = motion_timestamps - motion_timestamps[0]

    # Prepare data entry into backprojection function
    entry_data = {'scan_data': scan_data, 'platform_pos': platform_pos,
        'range_bins': file_data['range_bins'], 'scan_timestamps': scan_timestamps,
        'motion_timestamps': motion_timestamps, 'corner_reflector_pos': file_data['corner_reflector_pos']
    }

    # Visualize the data (to see alignment)
    visualize_data(entry_data)

    #file_opened = backproject_vectorize_real(entry_data, pixel_input, simulated=True, window=window_input)

    # Graph the data
    if graph:
        image_fig = plt.figure()
        image_ax = image_fig.add_subplot(111)
        h_img = image_ax.imshow(file_opened)
        plt.show()




if __name__ == '__main__':
    main()
