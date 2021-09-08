"""
testing.py defines time testing functions for TrackedObj over numpy ndarrays.

"""
from tracked_object import *

from aux_functions import *
from example_functions import *
import random
import time
import uuid


def meta_test_prov(funct, inputs = [(100, 100)], storage = './logs'):
    """
    inputs -> size of arrays?
    """
    print("STARTING PROVENANCE TEST")
    arrs = []
    # initialize inputs
    start = time.time()
    for input in inputs:
        arr = np.empty(input, dtype= object)
    
    end = time.time()
    print("initializing input array: {}".format(end-start))
    #reset provenance
    start = time.time()
    for i, arr in enumerate(arrs):
        arrs[i] = reset_array_prov(arr)
    end = time.time()
    print("resetting array provenance: {}".format(end - start))
    # run function
    start = time.time()
    output = funct(*arrs)
    
    end = time.time()
    print("running array function: {}".format(end - start))
    start = time.time()
    save_array_prov(output, storage)
    end = time.time()
    print('saving array provenance: {}'.format(end - start))

    print("FINISHED TEST")


        


def meta_test(funct, inputs = [(100, 100)]):
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


if __name__ == '__main__':
    meta_test_prov(test3)
    //meta_test(test3)
