import pickle
import bisect
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from numba import vectorize

datafile_name = "./data/5Points_1way_data.pkl"
# datafile_name = "./data/challenge_fun.pkl"

with open(datafile_name, 'rb') as file:
    data = pickle.load(file)

# print(data["range_bins"])
# last = 0
# for x in data["range_bins"][0] :
#     print(x-last)
#     last = x
# print("odjafoidsjfoi")
# print()
# print(data["scan_data"])
# print()
# print(data["platform_pos"])

data["scan_data"] = np.asarray(data["scan_data"])
data["platform_pos"] = np.asarray(data["platform_pos"])
data["range_bins"] = np.asarray(data["range_bins"])


x_loc_real = [6*a/120-3 for a in range(120)]
y_loc_real = [6*a/120-3 for a in range(120)]

pixels = [[b for b in range(120)] for a in range(120)]

@vectorize(['float32(float32, float32, float32, float32, float32, float32)'], target='cpu')
def distance(x_a, y_a, z_a, x_b, y_b, z_b) :
    return ( (x_a-x_b)**2 + (y_a-y_b)**2 + (z_a-z_b)**2 ) ** .5

start_time = time.time()

for x in range(len(x_loc_real)) :
    for y in range(len(y_loc_real)) :
        x_loc = x_loc_real[x]
        y_loc = y_loc_real[y]
        z_loc = 0

        total_dists = np.asarray([])

        platform_locs_full = data["platform_pos"]
        # print("PLATFORM LOCS DIMENSIONS")
        # print(len(platform_locs_full))
        # print(len(platform_locs_full[0]))
        platform_locs_x = platform_locs_full[:,0]
        platform_locs_y = platform_locs_full[:,1]
        platform_locs_z = platform_locs_full[:,2]
        x_locs = np.full(len(platform_locs_x), x_loc)
        y_locs = np.full(len(platform_locs_y), y_loc)
        z_locs = np.zeros(len(x_locs))
        
        total_dists = np.sqrt( np.square( (platform_locs_x-x_locs) ) + np.square( (platform_locs_y-y_locs) ) + np.square( (platform_locs_z-z_locs) ) )
        # total_dists = distance(platform_locs_x, platform_locs_y, platform_locs_z, x_loc, y_loc, z_loc)

        
        
        # x = np.square( (platform_locs_x-x_locs) )

        # for i in range(len(data["platform_pos"])) :
        #     platform_loc = data["platform_pos"][i]
        #     total_dists.append(math.sqrt( (platform_loc[0]-x_loc)**2 + (platform_loc[1]-y_loc)**2 + (platform_loc[2]-z_loc)**2 ) )
        #     # plane
        #     # result = data["range_bins"][0][closest_range_bin - 1]
        
        
        
        range_bins_list = []
        for i in range(len(total_dists)) :
            closest_range_bin_ind = bisect.bisect_left(data["range_bins"][0], total_dists[i]) -1 #TODO: see if its -1 ind or not
            # closest_range_bin_ind = int(round((total_dists[i] - data["range_bins"][0][0])/0.01845))
            # closest_range_bin_ind = math.floor((total_dists[i] - data["range_bins"][0][0])/0.01845)
            # print((total_dists[i] - data["range_bins"][0][0])/0.01845)

            scan_data = data["scan_data"][i][closest_range_bin_ind]
            range_bins_list.append(scan_data)
            
            # for s in range(scan_data) :
            #     scan_data[s] = np.abs(scan_data[s])
            # print("scan_data: "+str(scan_data))
        sum_range_bins = np.sum(range_bins_list)
        sum_range_bins = np.abs(sum_range_bins)
        
        pixels[y][x] = sum_range_bins

end_time = time.time()
elapsed_time = end_time-start_time
print("Done.")
print("Elapsed time: "+str(elapsed_time))
imgplot = plt.imshow(pixels, interpolation="gaussian")
plt.show()
