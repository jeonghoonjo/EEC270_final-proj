import os
import numpy as np

a = [[1, 2, 3, 4], [2, 3, 4, 5], [5, 6, 7, 8]]

min_t = np.amin(a)
max_t = np.amax(a)

for i in range(3):
    min_ = np.amin(a[i])
    max_ = np.amax(a[i])
    if min_ < min_t:
        min_t = min_
    if max_ > max_t:
        max_t = max_

print(min_t)
print(max_t)

for i in range(3):
    for j in range(4):
        a[i][j] -= min_t
        a[i][j] /= (max_t - min_t)
