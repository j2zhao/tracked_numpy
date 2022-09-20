
from typing import final
from compression_algs import *
from query_compression import *
from query_dslog import *
import os
import pickle
import numpy as np
import random
from query_compression import *
from query_dslog import *

import time

def compression_convert(folder1, folder2, num_steps, dfile, input2, images):
    final_shape = None
    for i in range(num_steps):
        array_file = os.path.join(folder1, 'step{}{}'.format(i, dfile))
        print(array_file)
        if dfile == '.pickle':
            with open(array_file, 'rb') as f:
                array = pickle.load(f)
        elif dfile == '.npy':
            array = np.load(array_file)
        
        final_shape = array.shape

        if i not in input2:
            ids = [1]
        else:
            ids = [1,2]
        if i not in images:
            raw_save(array, folder2, 'step{}_'.format(i), ids = ids, arrow = False)
        else:
            raw_save(array, folder2, 'step{}_'.format(i), ids = ids, image = False)
    return final_shape

def get_range(xsize, ysize, x, y):
    print(x)
    print(y)
    if x <= xsize:
        xstart = 0
        xmax = x -1
    else:
        xstart = random.randrange(0, x - xsize)
        xmax = xstart + xsize - 1
    if y <= ysize:
        ystart = 0
        ymax = y -1
    else:
        ystart = random.randrange(0, y - ysize)
        ymax = ystart + ysize - 1

    return (xstart, xmax), (ystart, ymax)

def make_compression(f2, f1, num_steps, folder_range = 20, input2 = [], images = []):
    for i in range(folder_range):
        folder2 = f2 + str(i)
        folder1 = f1 + str(i)
        try:
            shutil.rmtree(folder2)
        except OSError as e:
            pass
        os.mkdir(folder2)

        x, y = compression_convert(folder1, folder2, num_steps, dfile = '.pickle', input2, images)
        with open(os.path.join(folder2, 'x.pickle'), 'wb') as f:
            pickle.dump(x, f)
        with open(os.path.join(folder2, 'y.pickle'), 'wb') as f:
            pickle.dump(y, f)

if __name__ == '__main__':
    #folder1 = 'compression_tests_2/numpy_pipeline' + str(j)
    sizes = [(1, 1), (10, 1), (100, 1), (1000, 1), (1000, 10), (1000, 100)]
    experiments = [1, 10, 100, 1000, 10000, 100000]
    num_steps = 5
    for k in len(experiments):
        times = []
        xsize = sizes[k][0]
        ysize = size[k][1]
        experiment = experiments[k]
        for j in range(20):
            # get folder name and last size
            folder2 = 'storage/raw' + str(j)
            print(folder2)
            with open(os.path.join(folder2, 'x.pickle'), 'rb') as f:
                x = pickle.load(f)
            with open(os.path.join(folder2, 'y.pickle'), 'rb') as f:
                y = pickle.load(f)
            # get ranges and step names
            pranges = [get_range(xsize, ysize, x, y)]
            tnames = []
            for i in range(num_steps):
                tname = 'step{}_1'.format(i)
                tnames.append(tname)
            tnames.reverse()
            
            # get query results
            start = time.time()
            result = query_one2one(pranges, folder2, tnames, backwards = True, dtype = 'csv')
            end = time.time()
            times.append(end - start)
            #result = query_comp(pranges, folder2, tnames, absolute = False, merge = True, dtype = 'arrow')
        times = np.asarray(times)
        avg = np.average(times)
        std = np.std(times)
        print('finished experiment: {}'.format(experiment))
        print('average time: {}'.format(avg))
        print('std time: {}'.format(std))
        print(avg)
        np.save('query_results/raw_times{}.npy'.format(experiment), times)
    
