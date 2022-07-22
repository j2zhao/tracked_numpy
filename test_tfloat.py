from torch import conv2d
import numpy.core.tracked_float as annotate
import numpy as np
# from compression_tests.compression import compression
import numpy.core.tracked_float as tf
import time
import cv2
x = np.random.random((10, 10))
y = cv2.resize(x, (2, 2))
#print(dir(tf))
# x = np.random.random((10, 10)).astype(tf.tracked_float)
# tf.initialize(x, 1)
# y = np.ones((10, 10)).astype(tf.tracked_float)
# tf.initialize(y, 2)
# z = np.ones((10, 10)).astype(tf.tracked_float)
# tf.initialize(z, 3)

# A = np.random.random((3,3))
# B = np.dot(A, A.transpose())
# C = B.astype(tf.tracked_float)
# tf.initialize(C, 1)
# #x = np.reshape(x, (-1, ))
# #y = np.reshape(y, (-1, ))

# for i in range(3):
#     for j in range(3):
#         print(x[i, j].n)

# print('here')
# a= np.diff(x)
# for i in range(3):
#     for j in range(3):
#         print(a[i,j].provenance)
# print(type(a))
# if isinstance(a, np.ndarray):  
#     print(a.dtype)

# print(type(b))
# if isinstance(b, np.ndarray):  
#     print(b.dtype)
# print(a.shape)

# prov = compression(a)
# print(prov)
# np.save('/test2', x)
# print(z[0, 0].pnum)
# print(z[0, 0].num)
# print(z[0,0].num)
# print(z[0,0].num)
# print(z[0,0].num)
# print(z[0,0].num)
# print(z[0,0].num)
# print(z[0,0].num)
# print(z[0,0].num)
# print(z[0,0].num)
# print(z[0,0].num)
# z = np.sum(y, axis = 1, dtype=tf.tracked_float, initial=None)

# print(z[1].n)
# print(z[1].n)
# X = np.random.random((10, 10))

# def merge_column(X):
#     Y = np.zeros((X.shape[1]))
#     for i in range(X.shape[1]):
#         Y[i] = X[0, 0] + X[1, i]
#     return Y


# X = X.astype(annotate.annotated_double)
# annotate.initialize(X, 1)
# Y = np.sum(X, axis = 0)
# print(Y[0,0].provenance)