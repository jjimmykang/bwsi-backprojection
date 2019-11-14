import numpy as np
from control import unpack
import argparse
import pickle


def main():
    parser = argparse.ArgumentParser(description='batch unpacks data')
    parser.add_argument('data_dir', type=str, help='the directory for all of the scan datas')
    parser.add_argument('num_files', type=int, help='the number of files')
    args = parser.parse_args()

    data = []
    for i in range(1, args.num_files + 1):
        data.append(unpack(args.data_dir + '/scan_' + str(i) + '.txt'))

    np_data = np.asarray(data)
    print(list(data[1].keys()))

    master_dict = {'scan_data': [], 'timestamps': [], 'pulse_idx': [], 'range_bins': [], 'packet_idx': [], 'config': []}
    for i in data:
        master_dict['scan_data'].append(i['scan_data'])
        master_dict['timestamps'].append(i['timestamps'])
        master_dict['pulse_idx'].append(i['pulse_idx'])
        master_dict['range_bins'].append(i['range_bins'])
        master_dict['packet_idx'].append(i['packet_idx'])
        master_dict['config'].append(i['config'])






if __name__ == '__main__':
    main()
