
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


def get_range(xsize, ysize, x, y):
    if x < xsize:
        ysize = int(xsize/x)*ysize
    if y < ysize:
        xsize = int(ysize/y)*xsize

    if x <= xsize:
        xstart = 0
        xmax = x - 1
    else:
        xstart = random.randrange(0, x - xsize)
        xmax = xstart + xsize - 1
    if y <= ysize:
        ystart = 0
        ymax = y - 1
    else:
        ystart = random.randrange(0, y - ysize)
        ymax = ystart + ysize - 1

    return (xstart, xmax), (ystart, ymax)

if __name__ == '__main__':
    
    sizes = [(1, 1), (10, 1), (100, 1), (1000, 1), (1000, 10), (1000, 100)]
    experiments = [1, 10, 100, 1000, 10000, 100000]
    #experiments = [1]
    num_steps = 5
    for k in range(len(experiments)):
        times = []
        xsize = sizes[k][0]
        ysize = sizes[k][1]
        experiment = experiments[k]
        for j in range(20):
            # get folder name and last size
            folder2 = 'storage/dslog_gzi' + str(j)
            #folder2 = 'storage/turbo' + str(j)
            print(folder2)
            with open(os.path.join(folder2, 'x.pickle'), 'rb') as f:
                x = pickle.load(f)
            with open(os.path.join(folder2, 'y.pickle'), 'rb') as f:
                y = pickle.load(f)
            # get ranges and step names
            pranges = [get_range(xsize, ysize, x, y)]
            tnames = []
            for i in range(num_steps):
                tname = 'step{}_back1'.format(i)
                tnames.append(tname)
            tnames.reverse()
            # get query results
            start = time.time()
            query_comp(pranges, folder2, tnames, merge = True, dtype = 'arrow')
            #result = query_one2one(pranges, folder2, tnames, backwards = True, dtype = 'turbo')
            end = time.time()
            times.append(end - start)
            #result = query_comp(pranges, folder2, tnames, absolute = False, merge = True, dtype = 'arrow')
        times = np.asarray(times)
        avg = np.average(times)
        std = np.std(times)
        print('finished experiment: {}'.format(experiment))
        print('average time: {}'.format(avg))
        print('std time: {}'.format(std))
        np.save('query_results/dslog_gzip{}.npy'.format(experiment), times)
    
