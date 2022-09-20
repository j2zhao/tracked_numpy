
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
            raw_save(array, folder2, 'step{}_'.format(i), ids = ids, image = True)
    return final_shape

def get_range(xsize, ysize, x, y):
    xstart = random.randrange(0, x - xsize)     
    ystart = random.randrange(0, y - ysize)
    return (xstart, xstart + xsize - 1), (ystart, ystart + ysize - 1)


if __name__ == '__main__':
    times = []
    experiment = 0
    for j in range(100):
        #folder1 = 'compression_tests_2/numpy_pipeline' + str(j)
        folder1 = ''
        folder2 = 'storage/raw' + str(j)
        # try:
        #     shutil.rmtree(folder2)
        # except OSError as e:
        #     pass
        #os.mkdir(folder2)
        print(folder2)
        num_steps = 5
        input2 = []
        images = []
        dfile = '.pickle'
        xsize = 1
        ysize = 1
        if folder1 != '':
            x, y = compression_convert(folder1, folder2, num_steps, dfile, input2, images)
        else:
            x = 1000
            y = 100
        pranges = [get_range(xsize, ysize, x, y)]
        tnames = []
        for i in range(num_steps):
            tname = 'step{}_1'.format(i)
            tnames.append(tname)
        tnames.reverse()
        #result = query_comp(pranges, folder2, tnames, absolute = False, merge = True, dtype = 'arrow')
        #result = quer
        # y_invertedlist(pranges, folder2, tnames, dtype = 'arrow')
        #print(pranges)
        start = time.time()
        result = query_one2one(pranges, folder2, tnames, backwards = True, dtype = 'csv')
        end = time.time()
        times.append(end - start)
        ##result = query_invertedlist(pranges, folder2, tnames, dtype = 'arrow')
        print(result)
    times = np.asarray(times)
    avg = np.average(times)
    std = np.std(times)
    print('average time')
    print(avg)
    print(std)
    np.save('query_results/times{}.npy'.format(experiment), times)
    
