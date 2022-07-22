
from compression_algs import *
from query_compression import *
from query_dslog import *
import os
import pickle
import numpy as np
import random
from query_compression import *
from query_dslog import *
def compression_convert(folder1, folder2, num_steps, dfile, save_func, input2, images):
    for i in range(num_steps):
        array_file = os.path.join(folder1, 'step{}{}'.format(i, dfile))
        if dfile == '.pickle':
            with open(array_file, 'rb') as f:
                array = pickle.load(f)
        elif dfile == '.npy':
            array = np.load(array_file)
        
        if i not in input2:
            ids = [1]
        else:
            ids = [1,2]
        if i not in images:
            save_func(array, folder2, 'step{}'.format(i), ids = ids)
        else:
            save_func(array, folder2, 'step{}'.format(i), ids = ids, image = True)

def get_range(xsize, ysize, x, y):
    xstart = random.randrange(0, x - xsize)     
    ystart = random.randrange(0, y - ysize)
    return (xstart, xstart + xsize), (ystart, ystart + ysize)


if __name__ == '__main__':
    folder1 = ''
    folder2 = 'storage'
    save_func =  None
    num_steps = 1
    input2 = []
    images = []
    dfile = ''
    if folder1 != '':
        compression_convert(folder1, folder2, num_steps, dfile, save_func, input2, images)
    
    xsize = 3
    ysize = 3
    x = 10
    y = 10
    pranges = [get_range(xsize, ysize, x, y)]

    tnames = []
    for i in range(num_steps):
        tname = 'step{}_1'.format(i)
        tnames.append(tname)
    #result = query_comp(pranges, folder2, tnames, absolute = False, merge = True, dtype = 'arrow')
    #result = query_invertedlist(pranges, folder2, tnames, dtype = 'arrow')
    print(pranges)
    result = query_one2one(pranges, folder2, tnames, dtype = 'turbo')
    ##result = query_invertedlist(pranges, folder2, tnames, dtype = 'arrow')
    print(result)