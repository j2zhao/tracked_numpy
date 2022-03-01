import numpy as np
import numpy.core.tracked_float as tf

def test1(arr):
    # basic test
    #arr = np.random.random(arr_size).astype(tf.tracked_float)
    #tf.initialize(arr, 1)
    arr = np.negative(arr)
    
    return arr

def test2(arr, arr2):
    # test 2 arrays
    #arr = np.random.random(arr).astype(tf.tracked_float)
    #tf.initialize(arr, 1)
    #arr2 = np.random.random(arr2).astype(tf.tracked_float)
    #tf.initialize(arr2, 2)
    arr = arr + arr2
    return arr

def test3(arr):
    # test reduction
    #arr = np.random.random(arr).astype(tf.tracked_float)
    #tf.initialize(arr, 1)
    #arr = np.sum(arr, axis = 1, initial=None)
    arr2 = np.ones((arr.shape[1], ))
    arr = np.dot(arr, arr2)
    return arr

def test4(arr):
    #tests duplication
    #arr = np.random.random(arr).astype(tf.tracked_float)
    #tf.initialize(arr, 1)
    arr = np.tile(arr, (2, 2))
    return arr

def test5(arr):
    #tests duplication
    #arr = np.random.random(arr).astype(tf.tracked_float)
    #tf.initialize(arr, 1)
    arr = np.tile(arr, (3, 1))
    return arr

def test6(arr, arr2):
    #test matmul
    #arr = np.random.random(arr).astype(tf.tracked_float)
    #tf.initialize(arr, 1)
    #arr2 = np.random.random(arr2).astype(tf.tracked_float)
    #tf.initialize(arr2, 2)
    arr = np.dot(arr, arr2)
    return arr

def test7(arr, arr2):
    # test vector*vector
    #arr = np.random.random(arr_shape).astype(tf.tracked_float)
    #tf.initialize(arr, 1)
    #arr2 = np.random.random(arr_shape).astype(tf.tracked_float)
    #tf.initialize(arr2, 2)
    arr = np.reshape(arr, (arr.shape[0], ))
    arr2 = np.reshape(arr2, (arr.shape[0], ))
    arr = np.dot(arr, arr2)
    return arr

def test8(arr, arr2):
    # test matrix*vector
    #arr = np.random.random(arr).astype(tf.tracked_float)
    #tf.initialize(arr, 1)
    #arr2 = np.random.random(arr2).astype(tf.tracked_float)
    #tf.initialize(arr2, 2)
    np.reshape(arr2, (1000, ))
    arr = np.dot(arr, arr2)
    return arr

def test9_gen(arr_size = 1000000):
    arr = np.random.random(arr_size)
    return (arr,)

def test9_tf(arr):
    # Includes initalization
    arr = arr.astype(tf.tracked_float)
    tf.initialize(arr, 1)
    #if tracked_float:
    out = np.zeros(arr.shape).astype(tf.tracked_float)
    out[arr < 0.5] = arr[arr < 0.5]
    return out   

def test9(arr):
    # Includes initalization
    #if tracked_float:
    out = np.zeros(arr.shape)
    out[arr < 0.5] = arr[arr < 0.5]
    return out 

def test10_gen(arr_size = 1000000):
    x = np.arange(arr_size) 
    z = arr_size/2 
    std = arr_size/10
    y = np.random.normal(x, std)
    y = np.reshape(y, (arr_size,1))
    return (y, z)

def test10_tf(y, z):
    y = y.astype(tf.tracked_float)                                                             
    tf.initialize(y, 1)                                                                                                 
    out = np.zeros(y.shape).astype(tf.tracked_float)                                                                    
    out[y < z] = y[y < z]                                                                                               
    return out

def test10(y, z):
    out = np.zeros(y.shape)                                                                   
    out[y < z] = y[y < z]                                                                                               
    return out

def test11_gen(arr_shape = 1000000):
    arr = np.random.random((arr_shape, 1))
    return (arr,)

def test11_tf(arr):
    '''hist'''
    arr = arr.astype(tf.tracked_float)
    tf.initialize(arr, 1)
    arr = arr.reshape((-1,))

    arr_shape = arr.shape[0]
    out = np.zeros((4,)).astype(tf.tracked_float)
    arr_0 = np.zeros((arr_shape,)).astype(tf.tracked_float)
    arr_0[arr <= 0.25] = arr[arr < 0.25]
    arr_00 = np.zeros((arr_shape, )).astype(tf.tracked_float)
    out[0] = np.dot(arr_0, arr_00)

    arr_1 = np.zeros((arr_shape, )).astype(tf.tracked_float)
    index = np.logical_and((arr <= 0.5), (arr > 0.25))
    arr_1[index] = arr[index]
    arr_11 = np.zeros((arr_shape, )).astype(tf.tracked_float)
    out[1] = np.dot(arr_1, arr_11)

    arr_2 = np.zeros((arr_shape, )).astype(tf.tracked_float)
    index = np.logical_and((arr <= 0.75), (arr > 0.50))
    arr_2[index] = arr[index]
    arr_22 = np.zeros((arr_shape, )).astype(tf.tracked_float)
    out[2] = np.dot(arr_2, arr_22)

    arr_3 = np.zeros((arr_shape, )).astype(tf.tracked_float)
    index = (arr > 0.75)
    arr_3[index] = arr[index]
    arr_33 = np.zeros((arr_shape, )).astype(tf.tracked_float)
    out[3] = np.dot(arr_3, arr_33)

    return out, arr

def test11(arr):
    '''hist'''
    arr = arr.reshape((-1,))
    arr_shape = arr.shape[0]
    out = np.zeros((4,))
    arr_0 = np.zeros((arr_shape,))
    arr_0[arr <= 0.25] = arr[arr < 0.25]
    arr_00 = np.zeros((arr_shape, ))
    out[0] = np.dot(arr_0, arr_00)

    arr_1 = np.zeros((arr_shape, ))
    index = np.logical_and((arr <= 0.5), (arr > 0.25))
    arr_1[index] = arr[index]
    arr_11 = np.zeros((arr_shape, ))
    out[1] = np.dot(arr_1, arr_11)

    arr_2 = np.zeros((arr_shape, ))
    index = np.logical_and((arr <= 0.75), (arr > 0.50))
    arr_2[index] = arr[index]
    arr_22 = np.zeros((arr_shape, ))
    out[2] = np.dot(arr_2, arr_22)

    arr_3 = np.zeros((arr_shape, ))
    index = (arr > 0.75)
    arr_3[index] = arr[index]
    arr_33 = np.zeros((arr_shape, ))
    out[3] = np.dot(arr_3, arr_33)

    return out, arr

def test12_gen(arr_shape = 1000000):
    arr = np.sort(np.random.random((arr_shape,)))
    return (arr,)

def test12_tf(arr):
    '''hist'''
    arr = arr.astype(tf.tracked_float)
    arr = arr.reshape((-1, 1))
    tf.initialize(arr, 1)
    arr = arr.reshape((-1,))
    
    arr_shape = arr.shape[0]
    out = np.zeros((4,)).astype(tf.tracked_float)
    arr_0 = np.zeros((arr_shape,)).astype(tf.tracked_float)
    arr_0[arr <= 0.25] = arr[arr < 0.25]
    arr_00 = np.zeros((arr_shape, )).astype(tf.tracked_float)
    out[0] = np.dot(arr_0, arr_00)

    arr_1 = np.zeros((arr_shape, )).astype(tf.tracked_float)
    index = np.logical_and((arr <= 0.5), (arr > 0.25))
    arr_1[index] = arr[index]
    arr_11 = np.zeros((arr_shape, )).astype(tf.tracked_float)
    out[1] = np.dot(arr_1, arr_11)

    arr_2 = np.zeros((arr_shape, )).astype(tf.tracked_float)
    index = np.logical_and((arr <= 0.75), (arr > 0.50))
    arr_2[index] = arr[index]
    arr_22 = np.zeros((arr_shape, )).astype(tf.tracked_float)
    out[2] = np.dot(arr_2, arr_22)

    arr_3 = np.zeros((arr_shape, )).astype(tf.tracked_float)
    index = (arr > 0.75)
    arr_3[index] = arr[index]
    arr_33 = np.zeros((arr_shape, )).astype(tf.tracked_float)
    out[3] = np.dot(arr_3, arr_33)
    return out, arr

def test12(arr):
    '''hist'''
    arr = arr.reshape((-1,))
    arr_shape = arr.shape[0]
    out = np.zeros((4,))
    arr_0 = np.zeros((arr_shape,))
    arr_0[arr <= 0.25] = arr[arr < 0.25]
    arr_00 = np.zeros((arr_shape, ))
    out[0] = np.dot(arr_0, arr_00)

    arr_1 = np.zeros((arr_shape, ))
    index = np.logical_and((arr <= 0.5), (arr > 0.25))
    arr_1[index] = arr[index]
    arr_11 = np.zeros((arr_shape, ))
    out[1] = np.dot(arr_1, arr_11)

    arr_2 = np.zeros((arr_shape, ))
    index = np.logical_and((arr <= 0.75), (arr > 0.50))
    arr_2[index] = arr[index]
    arr_22 = np.zeros((arr_shape, ))
    out[2] = np.dot(arr_2, arr_22)

    arr_3 = np.zeros((arr_shape, ))
    index = (arr > 0.75)
    arr_3[index] = arr[index]
    arr_33 = np.zeros((arr_shape, ))
    out[3] = np.dot(arr_3, arr_33)

    return out, arr

def test13_gen(mnist = 'mnist_2.npy'):
    arr = np.load(mnist)
    arr = np.reshape(arr, (arr.shape[0], 1))
    return (arr,)

def test13_tf(arr):
    '''filter''' 
    arr.astype(tf.tracked_float)
    tf.initialize(arr, 1)
    out = np.zeros(arr.shape).astype(tf.tracked_float)
    out[arr > 0] = arr[arr > 0]
    return out

def test13_tf(arr):
    '''filter''' 
    out = np.zeros(arr.shape)
    out[arr > 0] = arr[arr > 0]
    return out