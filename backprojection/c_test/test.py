import numpy as np

test = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
test = test.reshape(2, 2, 3)
print(test.shape)
print(test)

test = test.reshape(2, -1)
print(test.shape)
print(test)
