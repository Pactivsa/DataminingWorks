import numpy as np

arr0 = np.array([[1, 2], [4, 5], [7, 8]])

arr1 = arr0[:,np.newaxis]
print(arr1, arr1.shape)

res = arr1 - arr0
print(res, res.shape)
'''
[[[1 2]]

 [[4 5]]

 [[7 8]]] (3, 1, 2)
 
[[[ 0  0]
  [-3 -3]
  [-6 -6]]

 [[ 3  3]
  [ 0  0]
  [-3 -3]]

 [[ 6  6]
  [ 3  3]
  [ 0  0]]] (3, 3, 2)
'''