import numpy as np
import pickle
import csv
import sys

MOTION_CAPTURE_FILENAME = sys.argv[1]
# RADAR_DATA_FILENAME = sys.argv[2]
PICKLE_DATA_FILENAME = sys.argv[2]

NAME_OF_OBJ = "S900"

# print(MOTION_CAPTURE_FILENAME)
# print(RADAR_DATA_FILENAME)

ALL_INFO_INDS = []
POS_INFO_INDS = []

POSITIONS = []

with open(MOTION_CAPTURE_FILENAME, newline='') as f:
    reader = csv.reader(f, delimiter=',', quotechar='|')
    r_ind = 0
    for row in reader :
        # print("next rows")
        for i in range(len(row)) :
            # print(i)
            if (r_ind == 3) and (row[i] == NAME_OF_OBJ) :
                ALL_INFO_INDS.append(i)
            if (r_ind == 5) and (i in ALL_INFO_INDS) and (row[i] == "Position") :
                POS_INFO_INDS.append(i)
        if r_ind >= 7 :
            POSITIONS.append( ((row[POS_INFO_INDS[0]], row[POS_INFO_INDS[1]], row[POS_INFO_INDS[2]]), row[1]) )
        r_ind += 1
        # input()
        # print(POSITIONS)

with open(PICKLE_DATA_FILENAME, 'wb') as p :
    pickle.dump(POSITIONS, p)