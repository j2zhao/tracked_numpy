import numpy as np
import numpy.core.tracked_float as tf

def test1(arr_size = (10, 100000)):
    # basic test
    arr = np.random.random(arr_size).astype(tf.tracked_float)
    tf.initialize(arr, 1)
    arr = np.negative(arr)
    
    return arr

def test2(arr = (10, 100000), arr2 = (10, 100000)):
    # test 2 arrays
    arr = np.random.random(arr).astype(tf.tracked_float)
    tf.initialize(arr, 1)
    arr2 = np.random.random(arr2).astype(tf.tracked_float)
    tf.initialize(arr2, 2)
    arr = arr + arr2
    return arr

def test3(arr = (1000, 1000)):
    # test reduction
    arr = np.random.random(arr).astype(tf.tracked_float)
    tf.initialize(arr, 1)
    arr = np.sum(arr, axis = 1, initial=None)
    return arr

def test4(arr = (10, 100000)):
    #tests duplication
    arr = np.random.random(arr).astype(tf.tracked_float)
    tf.initialize(arr, 1)
    arr = np.tile(arr, (2, 2))
    return arr

def test5(arr = (10, 100000)):
    #tests duplication
    arr = np.random.random(arr).astype(tf.tracked_float)
    tf.initialize(arr, 1)
    arr = np.tile(arr, (3, 1))
    return arr

def test6(arr = (1000, 1000), arr2 = (1000, 1000)):
    #test matmul
    arr = np.random.random(arr).astype(tf.tracked_float)
    tf.initialize(arr, 1)
    arr2 = np.random.random(arr2).astype(tf.tracked_float)
    tf.initialize(arr2, 2)
    arr = np.matmul(arr, arr2)
    return arr

def test7(arr_shape = (1000000, 1)):
    # test vector*vector
    arr = np.random.random(arr_shape).astype(tf.tracked_float)
    tf.initialize(arr, 1)
    arr2 = np.random.random(arr_shape).astype(tf.tracked_float)
    tf.initialize(arr2, 2)
    arr = np.reshape(arr, (arr_shape[0], ))
    arr2 = np.reshape(arr2, (arr_shape[0], ))
    arr = np.dot(arr, arr2)
    return arr

def test8(arr = (1000, 1000), arr2 = (1000, 1)):
    # test matrix*vector
    arr = np.random.random(arr).astype(tf.tracked_float)
    tf.initialize(arr, 1)
    arr2 = np.random.random(arr2).astype(tf.tracked_float)
    tf.initialize(arr2, 2)
    np.reshape(arr2, (1000, ))
    arr = np.dot(arr, arr2)
    return arr

def test9(arr = (1000000, 1)):
    # tests filters
    arr = np.random.random(arr).astype(tf.tracked_float)
    tf.initialize(arr, 1)
    out = np.zeros(arr.shape).astype(tf.tracked_float)
    out[arr < 0.5] = arr[arr < 0.5]
    return out    

def test10(arr = 1000000):
    x = np.arange(arr)                                                                                                  
    z = arr/2                                                                                                           
    std = arr/10                                                                                                        
    y = np.random.normal(x, std)                                                                                        
    y = np.reshape(y, (arr,1)).astype(tf.tracked_float)                                                                 
    tf.initialize(y, 1)                                                                                                 
    out = np.zeros(y.shape).astype(tf.tracked_float)                                                                    
    out[y < z] = y[y < z]                                                                                               
    return out

def test11(arr = 1000000, sorted = False):
    '''hist'''
    if sorted:
        arr = np.sort(np.random.random(arr)).astype(tf.tracked_float)
    else:
        arr = np.random.random(arr).astype(tf.tracked_float)

    tf.initialize(arr, 1)
    out = np.zeros((4,)).astype(tf.tracked_float) 
    for i in range(arr.shape[0]):
        if arr[i] <= 0.25:
            out[0] = out[0] + arr[i]
        elif arr[i] <= 0.5:
            out[1] = out[1] + arr[i]
        elif arr[i] <= 0.75:
            out[2] = out[2] + arr[i]
        else:
            out[2] = out[2] + arr[i]
    return out

def test12(arr = 1000000, sorted = True):
    '''hist'''
    if sorted:
        arr = np.sort(np.random.random(arr)).astype(tf.tracked_float)
    else:
        arr = np.random.random(arr).astype(tf.tracked_float)

    tf.initialize(arr, 1)
    out = np.zeros((4,)).astype(tf.tracked_float) 
    for i in range(arr.shape[0]):
        if arr[i].n <= 0.25:
            out[0] = out[0] + arr[i]
        elif arr[i].n <= 0.5:
            out[1] = out[1] + arr[i]
        elif arr[i].n <= 0.75:
            out[2] = out[2] + arr[i]
        else:
            out[2] = out[2] + arr[i]
    return out, arr # need arr for subzero

def test13(mnist = 'mnist.npy'):
    '''filter'''
    arr = np.load(mnist)
    tf.initialize(arr, 1)
    out = np.zeros(arr.shape).astype(tf.tracked_float)
    out[arr > 0] = arr[arr > 0]
    return out