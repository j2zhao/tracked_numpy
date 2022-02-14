
import numpy as np
import time
import os
import gzip
import json
import col_compression
from compression import compression
from compression_examples import *
import col_compression 
import time

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def aux(array):
    prov = np.empty(array.shape, dtype=object)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            prov[i, j] = array[i, j].provenance
    return prov
 
def raw_save(array, dir):
    dir = os.path.join(dir, str(time.time())+ '.npy')
    np.save(dir, array)

def gzip_save(array, dir):
    com = gzip.compress(array)
    dir = os.path.join(dir, str(time.time())+ '.npy')
    with open(dir, 'wb') as f:
        f.write(com)

def subzero_save(array, dir):
    """assume we already ran compression"""
    dir = os.path.join(dir, str(time.time())+ '.npy')
    with open(dir, 'wb') as f:
        json.dump(array,f)

def column_save(array, dir, temp_path, num_arrays, input_id = (1, 2)):

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
    dir = os.path.join(dir, str(time.time())+ '.npy')
    np.save(dir, array)

def comp_rel_save(array, dir):
    dir = os.path.join(dir, str(time.time())+ '.npy')
    array = compression(array, relative = True)
    np.save(dir, array)


if __name__ =="__main__":
    arr = test7()
    #arr = aux(arr)
    # arr = subzero.test1()
    # start = time.time()
    # dir = 'compressed/col'
    # if not os.path.isdir(dir):
    #     os.mkdir(dir)
    # temp_dir = 'compressed/temp_dir'
    # if not os.path.isdir(temp_dir):
    #     os.mkdir(temp_dir)
    # column_save(arr, dir, temp_dir, 1)
    # end = time.time()
    # size = get_size(dir)
    # print("Save time: {}".format(end - start))
    # print("size: {}".format(size))
    