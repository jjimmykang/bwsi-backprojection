intensities = [][]
for every x_loc:
    for every y_loc:
        total_dists = []
        ind = 0
        for every plane_pos in plane_locations:
            # plane_pos = (x_plane, y_plane, z_plane)
            # z_loc = 0
            total_dists[ind] = sqrt(dX^2 + dY^2 + dZ^2)
            ind+=1
        # plane_data is a list of (range[], intensity[], timestamp)
        LIST_SHIFT_PER_TIME = scale # defined configuration constant
        ind = 0
        for every dist in total_dists:
            time_shift = dist_to_time(dist-total_dists[ind-1])
            index_shift_val = time_shift/LIST_SHIFT_PER_TIME
            plane_data[ind] = shift_list( plane_data[ind] , index_shift_val )
            ind+=1
        total_data_sum = []
        for every data in plane_data :
            total_data_sum+=data
        intensity = total_data_sum #UNFINISHED
        intensities[x_loc][y_loc] = peak_intensity
#done
