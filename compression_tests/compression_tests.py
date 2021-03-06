
import numpy as np
import time
import os
import gzip
import json
import col_compression
from compression_tests.compression import compression
from compression_tests_2.compression_examples import *
import col_compression 
import subzero_functions
import time
import shutil
import pickle



def aux(array):
    if len(array.shape) == 0:
        array = np.reshape(array, (1, 1))
    
    if len(array.shape) == 1:
        array = np.reshape(array, (array.shape[0], 1))
    prov = np.empty(array.shape, dtype=object)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            prov[i, j] = array[i, j].provenance
    return prov
 
def raw_save(array, dir):
    dir = os.path.join(dir, 'raw.npy')
    np.save(dir, array)

def gzip_save(array, dir):
    com = gzip.compress(array)
    array = gzip.decompress(com)
    print(np.array(array))

    dir = os.path.join(dir, str(time.time())+ '.npy')
    with open(dir, 'wb') as f:
        f.write(com)

def gzip_2(file_name, new_file):
    with open(file_name, 'rb') as f_in:
        with gzip.open(new_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def subzero_save(array, dir):
    """assume we already ran compression"""
    dir = os.path.join(dir, 'raw.npy')
    with open(dir, 'wb') as f:
        pickle.dump(array,f)

def column_save(array, dir, temp_path, num_arrays, input_id = (1, 2)):
    if len(array.shape) == 0:
        array = np.reshape(array, (1, 1))
    
    if len(array.shape) == 1:
        array = np.reshape(array, (array.shape[0], 1))

    if num_arrays == 1:
        col_compression.to_column_1(array, temp_path, zeros = True)
    else:
        col_compression.to_column_2(array, temp_path, input_id, zeros = True)
    turbo_dir = "./turbo/Turbo-Range-Coder/turborc"
    turbo_param = "-20"
    if num_arrays == 1:
        file_names = ["x1.npy", "x2.npy", "y1.npy", "y2.npy"]
    else:
        file_names = ["x1.npy", "x2.npy", "y1.npy", "y2.npy", 'w1.npy', 'w2.npy', 'z1.npy', 'z2.npy']
    for name in file_names:
        p1 = os.path.join(temp_path, name)
        p2 = os.path.join(dir, name)
        command = " ".join([turbo_dir, turbo_param, p1, p2])
        os.system(command)

def comp_save(array, dir):
    array = compression(array, relative = False)
    dir = os.path.join(dir, 'raw.pickle')
    with open(dir, 'wb') as f:
        pickle.dump(array,f)

def comp_rel_save(array, dir):
    dir = os.path.join(dir, 'raw.pickle')
    array = compression(array, relative = True)
    with open(dir, 'wb') as f:
        pickle.dump(array,f)
    return array



if __name__ =="__main__":
    # gzip_2("compressed/raw.npy", "col/test.gzip")
    # _, arr = test11()
    # for i in range(4):
    #     print(arr[i].provenance)
    #arr = aux(arr)
    # # arr = subzero.test1()
    arr = subzero_functions.test6()
    dir = 'compressed'
    if not os.path.isdir(dir):
        os.mkdir(dir)
    temp_dir = 'temp'
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    subzero_save(arr, dir)
    #column_save(arr, dir, temp_dir, 1)
    #raw_save(arr, dir)
    #start = time.time()
    #arr = comp_rel_save(arr, dir)
    #print(arr)
    #gzip_2("compressed/raw.npy", "temp/test.gzip")
    #end = time.time()
    size = get_size(dir)
    print("size: {}".format(size))
    #size = get_size(temp_dir)
    #print("Save time: {}".format(end - start))
    #print("size: {}".format(size))
    