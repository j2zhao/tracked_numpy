"""
testing.py defines time testing functions for TrackedObj over numpy ndarrays.

"""
import numpy.core.tracked_float as tf
from aux_functions import *
from example_functions import *
import random
import time
import uuid


def meta_test_prov(funct, inputs = [(1, 100)], storage = './logs'):
    """
    inputs -> size of arrays?
    """
    print("STARTING PROVENANCE TEST")
    arrs = []
    # initialize inputs
    start = time.time()
    for input in inputs:
        arrs.append(np.random.random(input).astype(tf.tracked_float))
    
    #end = time.time()
    #print("initializing input array: {}".format(end-start))
    #reset provenance
    #start = time.time()
    for i, arr in enumerate(arrs):
        tf.initialize(arr, i)
    end = time.time()
    print("resetting array provenance: {}".format(end - start))
    # run function
    start = time.time()
    
    output = funct(*arrs)
    print(output[0,0])
    end = time.time()
    print("running array function: {}".format(end - start))

    # # start = time.time()
    # # save_array_prov(output, storage)
    # # end = time.time()
    # print('saving array provenance: {}'.format(end - start))

    # print("FINISHED TEST")
    return end-start


        


def meta_test(funct, inputs = [(10000, 10000)]):
    """
    inputs -> size of arrays?
    """
    print("STARTING REGULAR TEST")
    arrs = []
    # initialize inputs
    start = time.time()
    for input in inputs:
        arr = np.random.random(input)
        arrs.append(arr)
    
    end = time.time()
    print("initializing input array: {}".format(end-start))

    # run function
    start = time.time()
    output = funct(*arrs)
    end = time.time()
    print("running array function: {}".format(end - start))
    print("FINISHED TEST")
    return end-start


if __name__ == '__main__':
    meta_test_prov(test1, inputs= [(1, 100)])
    #meta_test(test1)
