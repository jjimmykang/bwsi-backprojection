import numpy
import pickle
import csv
import sys

MOTION_CAPTURE_FILENAME = sys.argv[1]
RADAR_DATA_FILENAME = sys.argv[2]
print(MOTION_CAPTURE_FILENAME)
print(RADAR_DATA_FILENAME)
with open(MOTION_CAPTURE_FILENAME, newline='') as f:
    reader = csv.reader(f, delimiter=' ', quotechar='|')
    print(type(reader))

