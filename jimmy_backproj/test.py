import numpy as np



return_array = np.zeros((120, 120))

encoded_data = np.zeros((120, 120))


position_map_3d = np.empty((120, 120, 2, 100))

# Generate coordinate system(120x120 array in which every value is a pair of coordinates(x,y))
rows = np.arange(4, -2, -6/120)
cols = np.arange(-3, 3, 6/120)

position_map = np.empty((len(rows), len(cols), 2), dtype=np.float64)
position_map[..., 0] = rows[:, None]
position_map[..., 1] = cols


position_map_3d[..., :] = position_map
