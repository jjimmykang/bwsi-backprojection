import pickle
import bisect
import math
import numpy as np
import matplotlib.pyplot as plt
import time

datafile_name = "./data/5Points_1way_data.pkl"

start_time = time.time()

with open(datafile_name, 'rb') as file:
    data = pickle.load(file)

data["scan_data"] = np.asarray(data["scan_data"])
data["platform_pos"] = np.asarray(data["platform_pos"])
data["range_bins"] = np.asarray(data["range_bins"])
range_bins_p = data["range_bins"][0]

pre_pixels = np.zeros((100,100))
pixels = np.zeros((100,100), dtype = np.complex128)

def bin_search(b) :
    return bisect.bisect_left(data["range_bins"][0], b) -1 #TODO: see if its -1 ind or not

def find_range_data(i, a) :
    return data["scan_data"][i][a]

x_loc_real = np.asarray([6*a/100-3 for a in range(100)])
z_loc_real = np.zeros(100)

x_loc_m = np.asarray([ x_loc_real for a in range(100) ])
y_loc_m = np.asarray([ x_loc_real for a in range(100) ])
y_loc_m = np.rot90(y_loc_m)
z_loc_m = np.asarray([ z_loc_real for a in range(100) ])

platform_locs_full = data["platform_pos"]

bin_search_v = np.vectorize(bin_search)
find_range_data_v = np.vectorize(find_range_data)

print("Starting...")
# start_time = time.time()

for i in range(len(data["platform_pos"])) :
    platform_loc_x = platform_locs_full[:,0][i]
    platform_loc_y = platform_locs_full[:,1][i]
    platform_loc_z = platform_locs_full[:,2][i]
    # platform_locs_x = np.full(len(x_loc_real), platform_loc_x)
    # platform_locs_y = np.full(len(y_loc_real), platform_loc_y)
    # platform_locs_z = np.full(len(z_loc_real), platform_loc_z)
    
    # total_dists = np.sqrt( ( platform_locs_x-x_loc_real )**2 + ( platform_locs_y-y_loc_real )**2 + ( platform_locs_z-z_loc_real )**2 )
    total_dists = np.sqrt( ( platform_loc_x-x_loc_m )**2 + ( platform_loc_y-y_loc_m )**2 + ( platform_loc_z-z_loc_m )**2 )
    # print(total_dists[0][0])
    # print(total_dists)
    pre_pixels = bin_search_v(total_dists)
    pre_pixels = find_range_data_v(i, pre_pixels)
    pre_pixels = pre_pixels
    pixels += pre_pixels

pixels = np.abs(pixels)

end_time = time.time()
elapsed_time = end_time-start_time
print("Done.")
print("Elapsed time: "+str(elapsed_time))
imgplot = plt.imshow(pixels) #, interpolation="gaussian")
plt.show()
