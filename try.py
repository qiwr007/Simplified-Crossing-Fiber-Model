import numpy as np

arr = np.array([[1,2,3,4],[5,6,7,8]])
brr = np.full((3,4), 0)
ind = np.array([[1,3,2,0],[3,2,1,0],[2,3,0,1]])

for i in range(arr.shape[0]):
    brr[i] = arr[i, ind[i]]