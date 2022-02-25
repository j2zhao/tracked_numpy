from prov_check import FunctionProvenance
import numpy as np
import random
import constants
import json
import pickle
import numpy.core.tracked_float as tf
import time

srange = [1, 100]
        
def generate_new_array(old_arrays):
    arrays = []
    for i in range(len(old_arrays)):
        d1, d2 = old_arrays[i].shape
        arrays.append(np.random.rand(d1, d2).astype(np.float64))
    return arrays

# given a csv of functions, generates a list of functions runs
def run_functions(arr_size, nfunc, args):

    func = getattr(np, nfunc)
    prov_obj = FunctionProvenance(log = './log/test.txt')

    for j in range(3):
        shape  = [(10, 20), (30, 40), (50, 60)]
        # arg = [(2, 100), (400, 3), (20, 150)]
        for i in range(3):
            arr = np.random.random(arr_size)
            #args_ = {'newshape': arg[i]}
            arg_dic, arr_tup, output, prov, t1, t2 = prov_obj.prov_function(func, [arr], args)
            
            if prov == None:
                provenance = prov_obj.compress_function(output)                
                prov_obj.add_prov(provenance, func.__name__, arg_dic, arr_tup)
                prov_obj.add_log(0, 0, func.__name__, arg_dic, arr_tup, provenance)
            
    
    tim = 0
    for i in range(100):
        arr = np.random.random((arr_size)).astype(np.float64)
        start = time.time()
        arg_dic, arr_tup, output, prov, t1, t2 = prov_obj.prov_function(func, [arr], args)
        end = time.time()
        tim += (end - start)
        print(t1)
        print(t2)
    return tim

def run_base_functions(arr_size, nfunc, args):
    arr = np.random.random(arr_size).astype(np.float64)
    func = getattr(np, nfunc)
    start = time.time()
    func(arr, **args)
    end = time.time()
    return end - start

if __name__ == '__main__':
    nfunc = 'reshape' # reshape
    args = {'newshape': (50, 2)} # ()
    #args = {}
    tim1 = 0
    tim2 = 0
    tim1 = run_functions((1, 100), nfunc, args)
    for i in range(100):
        tim2 += run_base_functions((1, 100), nfunc, args)
    print(tim1/100)
    print(tim2/100)