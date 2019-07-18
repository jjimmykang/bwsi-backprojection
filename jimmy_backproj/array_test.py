import numpy as np

array = np.asarray([1, 2, 3, 4, 5, 6])

i = np.searchsorted(array, 5.5, side='left')
print(i)
