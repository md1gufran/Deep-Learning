import numpy as np

x = np.array([1,2,3,4])
y = np.array([5,6,7,8])

var = np.array_split(x,2)
print(var)