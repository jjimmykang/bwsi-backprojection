import pickle
import bisect
import math
import numpy as np
import matplotlib.pyplot as plt

datafile_name = "./data/Mandrill_1way_data.pkl"
with open(datafile_name, 'rb') as file:
    data = pickle.load(file)

# print(data["range_bins"])
# print()
# print(data["scan_data"])
# print()
# print(data["platform_pos"])

x_loc_real = [6*a/120-3 for a in range(120)]
y_loc_real = [6*a/120-3 for a in range(120)]

pixels = [[b for b in range(120)] for a in range(120)]

for x in range(len(x_loc_real)) :
    for y in range(len(y_loc_real)) :
        x_loc = x_loc_real[x]
        y_loc = y_loc_real[y]
        z_loc = 0
        total_dists = []
        for i in range(len(data["platform_pos"])) :
            platform_loc = data["platform_pos"][i]
            total_dists.append(math.sqrt( (platform_loc[0]-x_loc)**2 + (platform_loc[1]-y_loc)**2 + (platform_loc[2]-z_loc)**2 ) )
            # plane
            # result = data["range_bins"][0][closest_range_bin - 1]
        range_bins_list = []
        for i in range(len(total_dists)) :
            # print(total_dists[i])
            closest_range_bin_ind = bisect.bisect_left(data["range_bins"][0], total_dists[i]) -1 #TODO: see if its -1 ind or not
            # # print(closest_range_bin_ind)
            # print(len(data["scan_data"][i]))
            scan_data = data["scan_data"][i][closest_range_bin_ind]
            range_bins_list.append(scan_data)
            
            # for s in range(scan_data) :
            #     scan_data[s] = np.abs(scan_data[s])
            # print("scan_data: "+str(scan_data))
        sum_range_bins = np.sum(range_bins_list)
        sum_range_bins = np.abs(sum_range_bins)
        
        pixels[x][y] = sum_range_bins

print("Done.")
imgplot = plt.imshow(pixels)
plt.show()