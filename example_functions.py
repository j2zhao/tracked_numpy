"""
example_functions.py contains example functions to run.
"""
# 1 or 2 matrix
# different types of operations
# just try repetition
import numpy.core.tracked_float as tf
import numpy as np

def test1(arr):
    # basic test
    for i in range(1):    
        arr = np.negative(arr)
    return arr

def test2(arr, arr2):
    # test 2 arrays
    for i in range(1):    
        a = arr + arr2
    return a

def test3(arr):
    # tests filters
    out = np.zeros(arr.shape).astype(tf.tracked_float)
    out[arr < 0.5] = arr[arr < 0.5]
    return out

def test4(arr):
    #tests reduction/long provenance
    arr = np.sum(arr, axis = 1, initial=None)
    #arr = np.reshape(arr,(1, arr.shape[0]))
    return arr

def test5(arr):
    #tests duplication
    arr = np.tile(arr, (2, 2))
    return arr



def test6(arr, arr2):
    #tests reduction/long provenance
    arr = np.dot(arr, arr2)
    #arr = np.reshape(arr,(1, arr.shape[0]))
    return arr