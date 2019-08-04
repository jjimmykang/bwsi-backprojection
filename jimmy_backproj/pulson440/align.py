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
import copy

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

def visualize_data(file_data, title, shift_distance, vmax):

    scan_data = file_data['scan_data']
    motion_timestamps = file_data['motion_timestamps']
    print('graphing!')

    # Plot
    plt.show()
    image_fig = plt.figure()

    image_fig.suptitle(title, fontsize=20)
    if not vmax == 0:
        h_img = plt.imshow(np.abs(scan_data),
         extent=(
            file_data['range_bins'][0],
            file_data['range_bins'][-1],
            file_data['scan_timestamps'][-1]-file_data['scan_timestamps'][0],
            0
        ),
        zorder=5, aspect='auto', vmax=vmax)
    else:
        h_img = plt.imshow(np.abs(scan_data),
         extent=(
            file_data['range_bins'][0],
            file_data['range_bins'][-1],
            file_data['scan_timestamps'][-1]-file_data['scan_timestamps'][0],
            0
        ),
        zorder=5, aspect='auto')
    plt.colorbar()

    ranges_1, ranges_2 = compute_ranges(file_data)
    #ranges_1 += shift_distance
    #ranges_2 += shift_distance

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


def regularize(array):
    return array - array[0]

def graph_signal(img):
    plt.show()
    image_fig = plt.figure()
    image_ax = image_fig.add_subplot(111)
    h_img = image_ax.imshow(img,
    zorder=5)
    plt.show()

def match_filter(entry_data, shift_mode, align_amt=None):

    motion_timestamps = entry_data['motion_timestamps']
    scan_timestamps = entry_data['scan_timestamps']
    platform_pos = entry_data['platform_pos']
    scan_data = entry_data['scan_data']
    scan_data_shape = scan_data.shape
    num_scans = scan_data_shape[0]
    range_bins = entry_data['range_bins']

    if shift_mode == 'manual':
        # SHIFTING CODE
        # Shift the data manually
        # Fetch align amount required
        if align_amt == None:
            align_amt = 0
        shift_time = align_amt

        # Translate the shifted time to indexes for each of the timestamp lists
        shift_motion_index = find_nearest(motion_timestamps, shift_time)

        # Slice the data arrays
        # This data is sliced because the parts that don't overlap must be eliminated
        # Motion/platform position data by the index
        motion_timestamps = motion_timestamps[shift_motion_index:]
        platform_pos = platform_pos[shift_motion_index:]

    elif shift_mode == 'automatic':
        # regularize
        motion_timestamps = regularize(motion_timestamps)
        scan_timestamps = regularize(scan_timestamps)

        # SHIFTING AUTOMATICALLY
        # Generate fake radar data from distance to reflectors
        # Proposed strength of each reflector

        range_1, range_2 = compute_ranges(entry_data)

        # Matrix for the signal map of the simulated reflectors
        reflector_signal = np.zeros((range_1.shape[0], scan_data_shape[1]))
        reflector_2_signal =np.zeros((range_2.shape[0], scan_data_shape[1]))

        # Generates the intensity graph of the reflector
        range_1_indexes = np.empty((range_1.shape))
        range_2_indexes = np.empty((range_2.shape))

        # Generate range 1 reflector intensity graph
        i = 0
        while i < range_1.shape[0]:
            range_1_indexes[i] = find_nearest(range_bins, range_1[i])
            i += 1

        signal_strength = 10000
        i = 0
        while i < range_1.shape[0]:
            reflector_signal[i, int(range_1_indexes[i])] = signal_strength
            i += 1

        # Generate same for range 2
        i = 0
        while i < range_2.shape[0]:
            range_2_indexes[i] = find_nearest(range_bins, range_2[i])
            i+= 1
        i = 0
        while i < range_2.shape[0]:
            reflector_2_signal[i, int(range_2_indexes[i])] = signal_strength
            i += 1

        reflector_signal = reflector_signal + reflector_2_signal

        # Create match filter

        scan_data_abs = np.abs(scan_data)
        regularized_point = (np.max(scan_data_abs) - np.min(scan_data_abs)) / 2
        scan_data_regularized = scan_data_abs - regularized_point
        lower_difference = reflector_signal.shape[0] - num_scans

        filtered_arr = np.empty((lower_difference))
        #graph_signal(scan_data_regularized)
        shift = 0

        while shift < lower_difference:
            multiplied_array = scan_data_regularized * reflector_signal[shift:-(lower_difference - shift)]
            mult_value = np.sum(multiplied_array)
            filtered_arr[shift] = mult_value
            shift += 1



        index_to_shift = find_nearest(filtered_arr, np.max(filtered_arr))

        motion_timestamps = motion_timestamps[index_to_shift:]
        platform_pos = platform_pos[index_to_shift:]
        motion_timestamps = regularize(motion_timestamps)

    # Re-regularize(because it got sliced)
    motion_timestamps = regularize(motion_timestamps)
    scan_timestamps = regularize(scan_timestamps)
    return {'scan_data': scan_data, 'platform_pos': platform_pos,
        'range_bins': entry_data['range_bins'], 'scan_timestamps': scan_timestamps,
        'motion_timestamps': motion_timestamps, 'corner_reflector_pos': entry_data['corner_reflector_pos']
    }


def main():
    # Argument Parser
    parser = argparse.ArgumentParser(description='align sar data')
    parser.add_argument('path_to_master', type=str, help='path to the master pickle file')
    parser.add_argument('file_name', type=str, help='name of the master pickle file ')
    parser.add_argument('first_cutoff', type=float, help='beginning cutoff in meters')
    parser.add_argument('last_cutoff', type=float, help='last cutoff in meters')
    parser.add_argument('--realx', type=float, help='Real world dimension x side')
    parser.add_argument('--realy', type=float, help='Real world dimension y side')
    parser.add_argument('--ppm', type=int, help='pixels per meter')
    parser.add_argument('--mode', type=int, help='toggle visualization mode')
    parser.add_argument('--shift', type=float, help='amount to shift the data by')
    parser.add_argument('-a', '--automatic', action='store_true', help='toggle automatic mode')
    parser.add_argument('-x', '--center_x', help='center for x')
    parser.add_argument('-y', '--center_y', help='center for y')
    parser.add_argument('-r', '--rangeshift', help='shift for range(m)')
    parser.add_argument('--rangecut', help='beginning meters to cut')
    parser.add_argument('--farcutoff', help='cut off the range bins that are far away.')
    parser.add_argument('--vmax', help='v-max on the display')
    args = parser.parse_args()

    # Command line argument processing
    graph = args.mode

    if args.realx == None:
        x_input = 6
    else:
        x_input = args.realx

    if args.realy == None:
        y_input = 6
    else:
        y_input = args.realy

    if args.ppm == None:
        ppm_input = 20
    else:
        ppm_input = args.ppm

    if not args.shift == None:
        align_amt = args.shift
    else:
        align_amt = 0

    if not args.rangecut == None:
        range_cut_dist = float(args.rangecut)
    else:
        range_cut_dist = 0

    if not args.farcutoff == None:
        far_cutoff_dist = float(args.farcutoff)
    else:
        far_cutoff_dist = 0

    if not args.vmax == None:
        vmax_input = float(args.vmax)
    else:
        vmax_input = 0


    file_data = open_file(args.path_to_master + '/' + args.file_name)


    # Initialize timestamps
    # Regularize at the same time
    motion_timestamps = file_data['motion_timestamps'] - file_data['motion_timestamps'][0]
    scan_timestamps = file_data['scan_timestamps'] - file_data['scan_timestamps'][0]
    scan_timestamps = scan_timestamps / 1000


    # Import data as variables
    # Adjust platform_pos so that the y and z are switched(treat z as height)
    platform_pos = copy.deepcopy(file_data['platform_pos'])


    platform_pos[..., 0] = file_data['platform_pos'][..., 1]
    platform_pos[..., 1] = file_data['platform_pos'][..., 0]


    corner_reflector_pos = copy.deepcopy(file_data['corner_reflector_pos'])
    corner_reflector_pos[..., 0] = file_data['corner_reflector_pos'][..., 1]
    corner_reflector_pos[..., 1] = file_data['corner_reflector_pos'][..., 0]


    range_bin_zeroed = copy.deepcopy(file_data['range_bins']) - file_data['range_bins'][0]

    # Cut off far range bins
    far_cutoff_index = find_nearest(range_bin_zeroed, far_cutoff_dist)
    scan_data = copy.deepcopy(file_data['scan_data'])[..., :-(far_cutoff_index+1)]
    range_bins = copy.deepcopy(file_data['range_bins'])[..., :-(far_cutoff_index+1)]

    # Set shape
    scan_data_shape = scan_data.shape
    # Get the number of scans
    num_scans = scan_data_shape[0]

    time_ratio = np.max(motion_timestamps) / np.max(scan_timestamps)

    # everything have the same index(same dimensional)
    ratio = (num_scans * time_ratio) / platform_pos.shape[0]
    scaled_platform_pos = np.empty((int(num_scans * time_ratio), 3))
    for i in np.arange(0, 3, 1):
        scaled_platform_pos[:, i] = np.interp(np.arange(0, int(num_scans * time_ratio), 1) / ratio, np.arange(0, platform_pos.shape[0], 1), platform_pos[:, i])

    motion_timestamps = np.interp(np.arange(0, int(num_scans * time_ratio), 1) / ratio, np.arange(0, motion_timestamps.shape[0], 1), motion_timestamps[:])
    platform_pos = scaled_platform_pos


    entry_data = {'scan_data': scan_data, 'platform_pos': platform_pos,
        'range_bins': range_bins, 'scan_timestamps': scan_timestamps,
        'motion_timestamps': motion_timestamps, 'corner_reflector_pos': corner_reflector_pos
    }


    # Manual range alignment
    if args.rangeshift != None:
        range_shift_distance = -float(args.rangeshift)
        range_bins = np.asarray(entry_data['range_bins']) + range_shift_distance
        entry_data['range_bins'] = range_bins

    else:
        range_shift_distance = 0


    if args.automatic == True:
        shift_mode = 'automatic'
        shift_amt = 0
    else:
        shift_mode = 'manual'
        shift_amt = args.shift

    entry_data = match_filter(entry_data, shift_mode, align_amt=shift_amt)
    motion_timestamps = entry_data['motion_timestamps']
    scan_timestamps = entry_data['scan_timestamps']
    platform_pos = entry_data['platform_pos']
    scan_data = entry_data['scan_data']
    scan_data_shape = scan_data.shape
    num_scans = scan_data_shape[0]
    range_bins = entry_data['range_bins']



    # CROPPING CODE
    # Cut ends off the data
    # in seconds
    cut_param = (args.first_cutoff, args.last_cutoff)

    motion_cut_param = (find_nearest(motion_timestamps, cut_param[0]), motion_timestamps.shape[0] - find_nearest(motion_timestamps, cut_param[1]))
    scan_cut_param = (find_nearest(scan_timestamps, cut_param[0]), scan_timestamps.shape[0] - find_nearest(scan_timestamps, cut_param[1]))


    motion_timestamps = motion_timestamps[motion_cut_param[0]:motion_cut_param[1]]
    scan_timestamps = scan_timestamps[scan_cut_param[0]:scan_cut_param[1]]

    platform_pos = platform_pos[motion_cut_param[0]:motion_cut_param[1]]
    scan_data = scan_data[scan_cut_param[0]:scan_cut_param[1]]

    motion_timestamps = regularize(motion_timestamps)
    scan_timestamps =  regularize(scan_timestamps)
    updated_num_scans = scan_data.shape[0]

    # Cut the rail noise(via user input)
    index_to_cut = find_nearest(range_bin_zeroed, range_cut_dist)
    scan_data[..., :index_to_cut] = 0


    # Scale data - make everything have the same length M
    # This, however, stll means that the motion data has a different time window than scan_data
    # TODO: merge the scalings so that we only have to do it once.
    ratio = updated_num_scans / platform_pos.shape[0]
    scaled_platform_pos = np.empty((updated_num_scans, 3))
    for i in np.arange(0, 3, 1):
        scaled_platform_pos[:, i] = np.interp(np.arange(0, updated_num_scans, 1) / ratio, np.arange(0, platform_pos.shape[0], 1), platform_pos[:, i])

    motion_timestamps = np.interp(np.arange(0, updated_num_scans, 1) / ratio, np.arange(0, motion_timestamps.shape[0], 1), motion_timestamps[:])
    platform_pos = scaled_platform_pos


    # Code to make the motion data have the same window as scan data
    time_cut = scan_timestamps[-1]
    index_motion_cut = find_nearest(motion_timestamps, time_cut)
    motion_timestamps = motion_timestamps[:index_motion_cut]
    platform_pos = platform_pos[:index_motion_cut]


    # Re-scale so the indexes are lined up again
    ratio = updated_num_scans / platform_pos.shape[0]
    scaled_platform_pos = np.empty((updated_num_scans, 3))
    for i in np.arange(0, 3, 1):
        scaled_platform_pos[:, i] = np.interp(np.arange(0, updated_num_scans, 1) / ratio, np.arange(0, platform_pos.shape[0], 1), platform_pos[:, i])

    motion_timestamps = np.interp(np.arange(0, updated_num_scans, 1) / ratio, np.arange(0, motion_timestamps.shape[0], 1), motion_timestamps[:])
    platform_pos = scaled_platform_pos

    # Prepare data entry into backprojection function
    # By this point, all data processing should be completed
    entry_data = {'scan_data': scan_data, 'platform_pos': platform_pos,
        'range_bins': range_bins, 'scan_timestamps': scan_timestamps,
        'motion_timestamps': motion_timestamps, 'corner_reflector_pos': corner_reflector_pos
    }
    print('corner_reflector_pos final:', entry_data['corner_reflector_pos'])

    # calculate average platform_pos so we can window properly
    average_platform_pos = (platform_pos[-1] + platform_pos[0]) / 2
    print('platform_pos[0]:', platform_pos[0])
    print('platform_pos[-1]:', platform_pos[-1])
    print('average_platform_pos:', average_platform_pos)

    # Also calculate recommended center
    recommended_center = (average_platform_pos + corner_reflector_pos[0] + corner_reflector_pos[1])/3
    print('RECOMMENDED CENTER:', recommended_center)


    print('graphing the data')
    # Graph the data
    if graph == 1:
        print('showing RTI')
        # 1 indicates RTI Plot
        visualize_data(entry_data, 'Final RTI', shift_distance=range_shift_distance, vmax=vmax_input)
    elif graph == 2:
        if args.center_x != None and args.center_y != None:
            center_input = (float(args.center_x), float(args.center_y))
        else:
            center_input = (0, 0)
        print('CENTER:', center_input)
        # 2 indicates Backprojected data
        backprojected_image = backproject_vectorize_real(entry_data, (x_input, y_input), ppm_input, simulated=True, center=center_input)
        print('backprojected_image.shape:', backprojected_image.shape)
        for i in np.arange(0, 3):
            backprojected_image = np.rot90(backprojected_image)
        image_fig = plt.figure()
        #image_ax = image_fig.add_subplot(111)
        if not vmax_input == 0:
            plt.imshow(backprojected_image, extent=(center_input[1]-(y_input/2), center_input[1]+(y_input/2), center_input[0]-(x_input/2), center_input[0]+(y_input/2)), vmax=vmax_input)
        else:
            plt.imshow(backprojected_image, extent=(center_input[1]-(y_input/2), center_input[1]+(x_input/2), center_input[0]-(x_input/2), center_input[0]+(x_input/2)))
        plt.colorbar()
        plt.show()
        with open(args.path_to_master + '/' + 'backprojected_data.pkl', 'wb') as p:
            pickle.dump(backprojected_image, p)
    elif graph == 3:
        print('scan_data:', np.max(np.log10(np.abs(scan_data))))
        ranges_1, ranges_2 = compute_ranges(file_data)
        print()
    # 0 indicates nothing visual happens



if __name__ == '__main__':
    main()
