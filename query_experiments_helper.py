
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

def compression_convert(folder1, folder2, num_steps, dfile, input2):
    final_shape = None
    for i in range(num_steps):
        array_file = os.path.join(folder1, 'step{}{}'.format(i, dfile))
        print(array_file)
        if dfile == '.pickle':
            with open(array_file, 'rb') as f:
                array = pickle.load(f)
        elif dfile == '.npy':
            array = np.load(array_file,allow_pickle=True)
        
        final_shape = array.shape

        if i not in input2:
            ids = [1]
        else:
            ids = [1,2]
        comp_rel_save(array, folder2, 'step{}_'.format(i), image = False, arrow = True, gzip = True)
        #column_save(array, folder2, 'step{}_'.format(i), temp_path = './temp', ids = ids)
        #raw_save(array, folder2, 'step{}_'.format(i), ids = ids, arrow = True)
        #gzip_save(array, folder2, 'step{}_'.format(i), ids = ids, arrow = True)
        #else:
            #raw_save(array, folder2, 'step{}_'.format(i), ids = ids, image = False, arrow=True)
    print(final_shape)
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
        x, y = compression_convert(folder1, folder2, num_steps, dfile = '.pickle', input2 = [])
        with open(os.path.join(folder2, 'x.pickle'), 'wb') as f:
            pickle.dump(x, f)
        with open(os.path.join(folder2, 'y.pickle'), 'wb') as f:
            pickle.dump(y, f)

def make_compression_image(f2, f1, num_steps):
    try:
        shutil.rmtree(f2)
    except OSError as e:
        pass
    os.mkdir(f2)
    x, y = compression_convert(f1, f2, num_steps, dfile = '.npy', input2 = [])
    with open(os.path.join(f2, 'x.pickle'), 'wb') as f:
        pickle.dump(x, f)
    with open(os.path.join(f2, 'y.pickle'), 'wb') as f:
        pickle.dump(y, f)

def make_compression_relational(f2, f1, num_steps):
    try:
        shutil.rmtree(f2)
    except OSError as e:
        pass
    os.mkdir(f2)
    x, y = compression_convert(f1, f2, num_steps, dfile = '.pickle', input2 = [0])
    with open(os.path.join(f2, 'x.pickle'), 'wb') as f:
        pickle.dump(x, f)
    with open(os.path.join(f2, 'y.pickle'), 'wb') as f:
        pickle.dump(y, f)

if __name__ == '__main__':
    folder1 = 'compression_tests_2/numpy_pipeline'
    folder2 = 'storage/dslog_giz'
    folder_range = list(range(20))
    #folder_range = []
    #make_compression_image(folder2, folder1, num_steps = 5)
    make_compression_numpy(folder2, folder1, 5, folder_range)