import numpy as np
import numpy.core.tracked_float as tf
import pandas as pd
from compression_array import array_compression
from relational_operations import *
import pickle


class DummyProv(object):
    def __init__(self, prov):
        self.provenance = prov

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
    arr = np.dot(arr, arr2)
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

def test9(arr_size = (1000000, 1)):
    # tests random filters
    arr = np.random.random(arr_size).astype(tf.tracked_float)
    tf.initialize(arr, 1)
    out = np.zeros(arr.shape).astype(tf.tracked_float)
    out[arr < 0.5] = arr[arr < 0.5]
    return out    

def test10(arr_size = 1000000):
    # test sorted filters
    x = np.arange(arr_size)                                                                                              
    z = arr_size/2                                                                                                           
    std = arr_size/10                                                                                                      
    y = np.random.normal(x, std)                                                                                        
    y = np.reshape(y, (arr_size,1)).astype(tf.tracked_float)                                                             
    tf.initialize(y, 1)                                                                                                 
    out = np.zeros(y.shape).astype(tf.tracked_float)                                                                    
    out[y < z] = y[y < z]                                                                                               
    return out

def test11(arr_shape = 1000000):
    '''random hist'''
    arr = np.random.random((arr_shape, 1)).astype(tf.tracked_float)
    tf.initialize(arr, 1)
    arr = arr.reshape((-1,))

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

    #for i in range(arr.shape[0]):


        # if arr[i,0].n <= 0.25:
        #     out[0] = out[0] + arr[i,0]
        # elif arr[i,0].n <= 0.5:
        #     out[1] = out[1] + arr[i,0]
        # elif arr[i,0].n <= 0.75:
        #     out[2] = out[2] + arr[i,0]
        # else:
        #     out[3] = out[3] + arr[i,0]
    return out, arr

def test12(arr_shape = 1000000, sorted = True):
    '''sorted hist'''
    arr = np.sort(np.random.random((arr_shape,)))
    arr = arr.astype(tf.tracked_float)
    arr = arr.reshape((-1, 1))
    tf.initialize(arr, 1)
    arr = arr.reshape((-1,))
    
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

def test13(mnist = 'mnist.npy'):
    '''image filter'''
    arr = np.load(mnist)
    arr = np.reshape(arr, (arr.shape[0], 1)).astype(tf.tracked_float)
    tf.initialize(arr, 1)
    out = np.zeros(arr.shape).astype(tf.tracked_float)
    out[arr > 0] = arr[arr > 0]
    return out

def test14(afile = './compression_tests_2/example_lime.npy', provenance = False):
    """
    LIME explainations
    """
    arr = np.load(afile)
    ar2 = np.zeros(arr.shape)
    ar2[arr > 0.5] = 1
    if provenance == False:
        return ar2
    else:
        prov = np.empty(arr.shape,dtype = object)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if ar2[i, j] == 1:
                    p = DummyProv([(1, i, j)])
                    prov[i, j] = p
                else:
                    p = DummyProv([])
                    prov[i, j] = p
        return prov

def test15(afile = './compression_tests_2/example_drise.npy', provenance = False):
    """
    DRISE explainations
    """
    arr = np.load(afile)
    ar2 = np.zeros(arr.shape)
    ar2[arr > 9] = 1
    if provenance == False:
        return ar2
    else:
        prov = np.empty(arr.shape,dtype = object)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if ar2[i, j] == 1:
                    p = DummyProv([(1, i, j)])
                    prov[i, j] = p
                else:
                    p = DummyProv([])
                    prov[i, j] = p
        return prov

def test16(data = './compression_tests_2/group_by_pandas.pickle', col_name = 'startYear', agg_name = 'isAdult'):
    """
    Groupby UnSorted
    """
    #data = pd.read_csv(data)
    with open(data, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    print(data.head(10))
    data = groupby_prov(data, col_name, agg_name)
    return data

def test17(data = './compression_tests_2/group_by_pandas.pickle', col_name = 'startYear', agg_name = 'isAdult'):
    """
    Groupby Sorted
    """
    #data = pd.read_csv(data)
    with open(data, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        data = data.head(1000000)
        data = data.sort_values(by='startYear', ignore_index= True)
    data = groupby_prov(data, col_name, agg_name)
    return data

def test18(data_left = './compression_tests_2/left_join_pandas.pickle', data_right = './compression_tests_2/right_join_pandas.pickle', column1 = 'tconst', column2 = 'tconst'):
    """
    Join Sorted
    """
    #df1 = pd.read_csv(data_left)
    #df2 = pd.read_csv(data_right)
    with open(data_left, 'rb') as f:
        df1 = pickle.load(f, encoding='latin1')
    with open(data_right, 'rb') as f:
        df2 = pickle.load(f, encoding='latin1')
    data = join_prov(df1, df2, column1, column2)
    return data

def test19(data_left = './compression_tests_2/left_join_pandas.pickle', data_right = './compression_tests_2/right_join_pandas.pickle', column1 = 'tconst', column2 = 'parentTconst'):
    """
    Join UnSorted
    """
    with open(data_left, 'rb') as f:
        df1 = pickle.load(f, encoding='latin1')
    with open(data_right, 'rb') as f:
        df2 = pickle.load(f, encoding='latin1')
    data = join_prov(df1, df2, column1, column2)
    return data

if __name__ == '__main__':
    arr = test18()
    print(arr)