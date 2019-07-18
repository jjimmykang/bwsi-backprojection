import numpy as np

def backproject_vectorize_c(data, dimensions, num_scans):
    '''Backprojects
    Arguments:
        data(dict)
            a dictionary of the data
        dimensions(tuple)
            the resolution of the data (x, y, z)
        num_scans(int)
            the number of scans

    Returns:
        return_array(array)
            x x y array that represents the data
    '''
    x_pixels = dimensions[0]
    y_pixels = dimensions[1]
    z_pixels = dimensions[2]

    # Generate needed objects
    data_array = np.asarray(data['scan_data'])
    return_array = np.empty((x_pixels, y_pixels))
    range_bins = np.asarray(data['range_bins'])
    encoded_data = np.empty((x_pixels, y_pixels))
    linear = True

    # Generate coordinate system(120x120 array in which every value is a pair of coordinates(x,y))
    cols = np.arange(-3, 3, 6/y_pixels)
    rows = np.arange(3, -3, -6/x_pixels)
    zetas = z_pixels

    position_map = np.empty((len(rows), len(cols), 3), dtype=np.float32)
    position_map[..., 0] = cols
    position_map[..., 1] = rows[:, None]
    position_map[..., 2] = zetas


    # Broadcast the position map onto a 100x120x120x2 array. It's one hundred of the array to individually calculate the distance for.
    # The first index is the number of pulse.
    # The rest of the array is dimension (120x120x2), representing ordered pairs.
    position_map_3d = np.empty((num_scans, x_pixels, y_pixels, 3))
    position_map_3d[:] = position_map

    # This code processes the platform data from the radar
    # Convert platform positions into 2d
    platform_pos = np.asarray(data['platform_pos'])

    '''
    # Get rid of the z coordinates
    platform_pos_2d = []
    for location in platform_pos:
        platform_pos_2d.append([location[0], location[1]])
    platform_pos_2d = np.asarray(platform_pos_2d)
    '''

    # Convert platform_pos_2d into a 100x120x120x2 array to overlay over position map
    # It's basically the identical layer throughout the 100 pulses
    platform_pos_3d = np.empty((num_scans, x_pixels, y_pixels, 3))
    platform_pos_3d[:, :, :, :] = platform_pos[:, None, None, :]

    # With the platform_map_3d, and position_map_3d, we can generate the distance lookup table.
    distance_lookup_table = np.empty((num_scans, x_pixels, y_pixels))
    distance_lookup_table = np.sqrt(np.power((position_map_3d[..., 0] - platform_pos_3d[..., 0]), 2) + \
                                    np.power((position_map_3d[..., 1] - platform_pos_3d[..., 1]), 2) + \
                                    np.power((position_map_3d[..., 2] - platform_pos_3d[..., 2]), 2))


    #Utilize the distance LUT to generate signal map.
    flattened_distance_lookup = np.reshape(distance_lookup_table, (num_scans, -1))
    signal_matrix = np.empty((num_scans, x_pixels * y_pixels), dtype=np.complex128)
    cdef int radar_counter = 0

    print('flattened_distance_lookup.shape', flattened_distance_lookup.shape)
    print('range_bins.shape', range_bins)
    print('data_array.shape', data_array.shape)
    while radar_counter < num_scans:
        signal_matrix[radar_counter] = np.interp(flattened_distance_lookup[radar_counter], range_bins, data_array[radar_counter])
        radar_counter += 1

    signal_matrix = np.abs(np.sum(signal_matrix, axis=0))
    signal_matrix = np.reshape(signal_matrix, (x_pixels, y_pixels))

    #strength_array = np.transpose(strength_array)
    #array = np.asarray([[1, 2, 3], [4, 5, 6]])
    #print(speed.speed_func(array))
    signal_matrix = 
    return signal_matrix
