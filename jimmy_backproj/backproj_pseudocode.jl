s1. 100x14400 - strength array 
2. reshape each layer from distance array
3. radar_counter = 0
  while radar_counter < 100:
    strength[radar_counter] = np.interp(distance, range_bin, data_array[radar_counter])

4. strength array is updated
5. flatten strength array down along the pulse side
