import numpy as np

a = np.array([1, 2, 3])
b = np.array([5, 6])

print(np.concatenate((a, b), axis=0))
print(len(a))