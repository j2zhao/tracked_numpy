import readline

from torch import float64
from prov_check import FunctionProvenance
import csv
import numpy as np
import sklearn
import random
import constants
import json
import pickle
#from precapture_args import *
srange = [1, 100]

# run function and saves to update later
def prov_function():
    pass


# run function and updates right away
def prov_function2():
    pass

# generates a list of function given arguments 
def iterate_parameters(pattern):
    index = random.randrange(len(pattern))
    row = pattern[index]
    n = len(row)
    arrs = row[1]
    if n > 2:
        other_args = row[2]
    else:
        other_args = {}


    dict_args = {}
    for tup in arrs:
        for i in tup:
            if i not in dict_args:
                if isinstance(i, int):
                    dict_args[i] = i
                else:
                    dict_args[i] = random.randrange(srange[0], srange[1])
                    prob = random.random()
    arrays = []
    for tup in arrs:
        d = []
        for i in range(len(tup)):
            d.append(dict_args[tup[i]])
        arrays.append(np.random.random(d).astype(np.float64))

    other = {}
    # other['axis'] = random.randrange(2)
    # other['obj']= random.randrange(dict_args[tup[other['axis']]])
    for key, val in other_args.items():
        index = random.randrange(len(val))
        other[key] = val[index]
        if val[index] == "custom":
            other[key] = random.randrange(dict_args[tup[1]])
        # #other[key] = (d1, 10)
        # if index == 2:
        #     other[key] = (dict_args['b'], dict_args['a'])
    return arrays, other
        
def generate_new_array(old_arrays):
    arrays = []
    for i in range(len(old_arrays)):
        #d1, d2 = old_arrays[i].shape
        arrays.append(np.random.random(old_arrays[i].shape).astype(np.float64))
    return arrays

# run one run of experiments
def run(provenance, func, arrays, args, repetition, log):
    prov_obj = FunctionProvenance(log = log, saved_prov=provenance)
    for i in range(repetition):
        print('Function {} Run {}'.format(func.__name__, i))
        if i != 0:
            arrays = generate_new_array(arrays)
        arg_dic, arr_tup, output, prov, t1, t2 = prov_obj.prov_function(func, arrays, args)
        if t2 == 0:
            print('type: general')
        elif t2 == 1:
            print('type: specfic')
        else:
            print('type: None')
        
        if t1 == 0:
            print('No Match')
        elif t1 == 1:
            print('Not enough iterations')
        elif t1 == 2:
            print('Pattern Unknown')
        elif t1 == 3:
            print('Pattern Found')
        
        if prov == None:
            provenance = prov_obj.compress_function(output)
            print(provenance)
            prov_obj.add_prov(provenance, func.__name__, arg_dic, arr_tup)
            #print(prov_obj.prov)
            prov_obj.add_log(0, 0, func.__name__, arg_dic, arr_tup, provenance)
    
    # for i in range(len(prov_obj.prov[func.__name__]['provs'])):
    #     print(prov_obj.prov[func.__name__]['provs'][i][1])
    print(prov_obj.prov[func.__name__]['provs'])
    return prov_obj.prov

# given a csv of functions, generates a list of functions runs
def run_functions(file = 'functions.csv', prov_save = 'provenance.pickle', rep = 3, num = 20):
    with open(file, 'r') as f:
        row = f.readline()
        pattern = []
        while row:
            row = json.loads(row)
            nfunc = row[0]
            pattern.append(row)
            row = row = f.readline()
        func = getattr(np, nfunc)
        provenance = None
        # with open(prov_save, 'r') as f:
        #     provenance =  pickle.load(f)
        for i in range(num):
            arrays, args = iterate_parameters(pattern)
            for array in arrays:
                print(array.shape)
            print(args)
            provenance = run(provenance, func, arrays, args, rep, './log/test.txt')
            with open(prov_save, 'wb') as prov:
                pickle.dump(provenance, prov)

if __name__ == '__main__':
    run_functions()