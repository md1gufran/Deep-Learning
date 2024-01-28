import numpy as np

var = np.array([1,2,3,4])
var1 = np.array([5,7,8,9])

#if we have different length of aaray then python will check whether broadcasting is possible or not
final = var + var1
print(final)