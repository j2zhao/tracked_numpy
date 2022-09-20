
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

def make_compression_numpy(f2, f1, num_steps, folder_range):
    for i in folder_range:
        folder2 = f2 + str(i)
        folder1 = f1 + str(i)
        try:
            shutil.rmtree(folder2)
        except OSError as e:
            pass
        os.mkdir(folder2)
        x, y = compression_convert(folder1, folder2, num_steps, dfile = '.pickle', input2 = [], images = [])
        with open(os.path.join(folder2, 'x.pickle'), 'wb') as f:
            pickle.dump(x, f)
        with open(os.path.join(folder2, 'y.pickle'), 'wb') as f:
            pickle.dump(y, f)

if __name__ == '__main__':
    folder1 = 'compression_tests_2/numpy_pipeline'
    folder2 = 'storage/raw'
    folder_range = list(range(20))
    make_compression_numpy(folder1, folder1, num_steps = 5, folder_range = folder_range)