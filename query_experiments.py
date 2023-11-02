
from compression_algs import *
from query_compression import *
from query_dslog import *
import os
import pickle
import numpy as np
import random
from query_compression import *
from query_dslog import *
import math
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


def query_experiments_numpy(shape, sizes, experiments, num_steps, num_exp, save_name = 'query_results_10/numpy_raw_results', folder_name = 'storage_10/numpy_no_pq', forward = False):
    for k in range(len(experiments)):
        times = []
        xsize = sizes[k][0]
        ysize = sizes[k][1]
        experiment = experiments[k]
        j = 0
        print(experiment)
        for j in range(num_exp):
            print(j)
            # get folder name and last size
            folder2 = folder_name + str(j)
            x = shape[0]
            y = shape[1]
            # get ranges and step names
            pranges = [get_range(xsize, ysize, x, y)]
            print(pranges)
            tnames = []
            for i in range(num_steps):
                if not forward:
                    tname = 'step{}_1'.format(i)
                else:
                    tname = 'step{}_for1'.format(i)
                tnames.append(tname)
            # get query results
            start = time.time()
            #result = query_comp(pranges, folder2, tnames, backward = False, merge = False, dtype = 'arrow')
            #result = query_one2one(pranges, folder2, tnames, backwards = False, dtype = 'turbo')
            if not forward:
                result = query_one2one_select(pranges, folder2, tnames, backwards = False, dtype = 'arrow')
            # elif not forward:
            #     result = query_one2one_select(pranges, folder2, tnames, backwards = False, dtype = 'arrow')
            else:
                result = query_comp(pranges, folder2, tnames, backward = False, merge = False, dtype = 'arrow')
            end = time.time()
            times.append(end - start)
                
        times = np.asarray(times)
        #np.save(save_name + '{}.npy'.format(experiment), times)

def query_experiemnts_pipeline(shape = [1080, 1920], folder2 = 'storage_pipeline/image_dslog'):
    #experiments = [0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
    experiments = [0.001, 0.01, 0.2, 0.8, 1]
    #sizes = [(int(math.sqrt(s)*shape[0]), int(math.sqrt(s)*shape[1])) for s in experiments]
    sizes = [(int(shape[0]), int(s*shape[1])) for s in experiments]
    experiments = [0, 1, 20, 80, 100]
    for k in range(len(experiments)):
        xsize = sizes[k][0]
        ysize = sizes[k][1]
        experiment = experiments[k]
        print(experiment)
        # get folder name and last size
        x = shape[0]
        y = shape[1]
        while True:
            # get ranges and step names
            pranges = [get_range(xsize, ysize, x, y)]
            print(pranges)
            tnames = []
            for i in range(5):
                tname = 'step{}_1'.format(i)
                tnames.append(tname)
            #tnames = ['step0_1']w
            #tnames.reverse()
            # get query results
            start = time.time()
            result = query_comp_join(pranges, folder2, tnames, backward = False, merge = True, dtype = 'arrow')
            #result = query_one2one(pranges, folder2, tnames, backwards = False, dtype = 'arrow')
            #print(result)
            end = time.time()
            #raise ValueError()
            if len(result) != 0:
                print('finished experiment: {}'.format(experiment))
                print(end - start)
                break
                
# def query_experiemnts_pipeline(shape = [1080, 1920], folder2 = 'storage_pipeline/image_dslog'):
#     pass


if __name__ == '__main__':
    #query_experiments_numpy(shape = [1000, 100], sizes = [(1, 1), (10, 1), (100, 1), (1000, 1), (1000, 10), (1000, 100)], \
    #    experiments = [1, 10, 100, 1000, 10000, 100000], num_steps = 5, num_exp = 20, save_name = 'query_results_5/numpy_new_pq_results', folder_name = 'storage_5/numpy_pq', forward = False)
    # query_experiments_numpy(shape = [1000, 100], sizes = [(1000, 100)], 
    #      experiments = [100000], num_steps = 5, num_exp = 20, save_name = 'query_results_5/numpy_dslog_merge_results', folder_name = 'storage_5/numpy_dslog', forward = True)
    #query_experiemnts_pipeline(shape = [1080, 1920], folder2 = './storage_pipeline/storage_image_compression/image_dslog')
    #query_experiemnts_pipeline(shape = [9, 9044976], folder2 = 'storage_pipeline/storage_relational_compression/relational_pq')
    query_experiemnts_pipeline(shape = [1080, 1920], folder2 = 'storage_pipeline/storage_resnet_compression/resnet_pq')
