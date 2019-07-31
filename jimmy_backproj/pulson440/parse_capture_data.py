import numpy as np
import pickle
import csv
import sys

MOTION_CAPTURE_FILENAME = sys.argv[1]
# RADAR_DATA_FILENAME = sys.argv[2]
PICKLE_DATA_FILENAME = sys.argv[2]
CAPTURE_TYPE = sys.argv[3]

NAME_OF_OBJ = CAPTURE_TYPE


# print(MOTION_CAPTURE_FILENAME)
# print(RADAR_DATA_FILENAME)

ALL_INFO_INDS = []
POS_INFO_INDS = []

POSITIONS = []

#PARSING POSITION DATA FOR MOCAP
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
            if row[POS_INFO_INDS[0]] != "" :
                POSITIONS.append( ((float(row[POS_INFO_INDS[0]]), float(row[POS_INFO_INDS[2]]), float(row[POS_INFO_INDS[1]])), float(row[1])) )
            else :
                POSITIONS.append( ((np.nan, np.nan, np.nan), float(row[1])) )
        r_ind += 1
        # input()
        # print(POSITIONS)

with open(PICKLE_DATA_FILENAME, 'wb') as p :
    print(POSITIONS)
    pickle.dump(POSITIONS, p)

# #PARSING DATA FOR RADAR
# def unpack(scan_data_filename):
#     with open(scan_data_filename, 'rb') as f:
#         # Read configuration part of data
#         config = read_config_data(f)
#         # Compute range bins in datas
#         scan_start_time = float(config['scan_start'])
#         start_range = SPEED_OF_LIGHT * ((scan_start_time * 1e-12) - DT_0 * 1e-9) / 2
#         # Initialize container for unpacked data
#         data = dict()
#         data = {'scan_data': [],
#                 'timestamps': [],
#                 'pulse_idx': None,
#                 'range_bins': None,
#                 'packet_idx': [],
#                 'config': config}
#         single_scan_data = []
#         packet_count = 0
#         pulse_count = 0
#         # Read data
#         while True:
#             # Read a single data packet and break loop if not a complete packet (in terms of size)
#             packet = f.read(MAX_SCAN_INFO_PACKET_SIZE)
#             if len(packet) < MAX_SCAN_INFO_PACKET_SIZE:
#                 break
#             # Get information from first packet about how scans are stored and range bins collected
#             if packet_count == 0:
#                 num_range_bins = np.frombuffer(packet[44:48], dtype='>u4')[0]
#                 num_packets_per_scan = np.frombuffer(packet[50:52], dtype='>u2')[0]
#                 drange_bins = SPEED_OF_LIGHT * T_BIN * 1e-9 / 2
#                 range_bins = (start_range + drange_bins * np.arange(0, num_range_bins, 1))

#             # Number of samples in current packet, timestamp, and packet index
#             num_samples = np.frombuffer(packet[42:44], dtype='>u2')[0]
#             timestamp = np.frombuffer(packet[8:12], dtype='>u4')[0]
#             data['packet_idx'].append(np.frombuffer(packet[48:50], dtype='>u2')[0])

#             # Extract radar data samples from current packet; process last packet within a scan
#             # seperately to get all data
#             packet_data = np.frombuffer(packet[52:(52 + 4 * num_samples)], dtype='>i4')
#             single_scan_data.append(packet_data)
#             packet_count += 1

#             if packet_count % num_packets_per_scan == 0:
#                 data['scan_data'].append(np.concatenate(single_scan_data))
#                 data['timestamps'].append(timestamp)
#                 single_scan_data = []
#                 pulse_count += 1

#         # Add last partial scan if present
#         if single_scan_data:
#             single_scan_data = np.concatenate(single_scan_data)
#             num_pad = data['scan_data'][0].size - single_scan_data.size
#             single_scan_data = np.pad(single_scan_data, (0, num_pad), 'constant', constant_values=0)
#             data['scan_data'].append(single_scan_data)
#             data['timestamps'].append(timestamp)
#             pulse_count += 1

#         # Stack scan data into 2-D array (rows -> pulses, columns -> range bins)
#         data['scan_data'] = np.stack(data['scan_data'])

#         # Finalize entries in data
#         data['timestamps'] = np.asarray(data['timestamps'])
#         data['pulse_idx'] = np.arange(0, pulse_count)
#         data['range_bins'] = range_bins
#         data['packet_idx'] = np.asarray(data['packet_idx'])

#         print(data['scan_data'])

#         return data


# #CALCULATING START/END POINTS FOR DATA


# #ALIGNING DATA


# #RESULT
