import numpy as np
import pickle
import csv
import sys

def extract_capture_data(PICKLE_DATA_FILENAME):
    with open(PICKLE_DATA_FILENAME, 'rb') as p :
        a = pickle.load(p)
        platform_pos = []
        motion_timestamps = []
        for x in a :
            platform_pos.append([x[0][0], x[0][1], x[0][2]])
            motion_timestamps.append(x[1])
        # print(platform_pos)
        # print(motion_timestamps)
        return (platform_pos, motion_timestamps)

extract_capture_data("./mocap_data/trial_003.pkl")