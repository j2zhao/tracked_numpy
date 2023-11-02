
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
        #print(i)
        array_file = os.path.join(folder1, 'step{}{}'.format(i + 1, dfile))
        #print(array_file)
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
        
        #raw_save(array, folder2[0], 'step{}_'.format(i), ids = ids, arrow = False)
        #raw_save(array, folder2[1], 'step{}_'.format(i), ids = ids, arrow = True)
        #gzip_save(array, folder2[2], 'step{}_'.format(i), ids = ids, arrow = True)
        #column_save(array, folder2[3], 'step{}_'.format(i), temp_path = './temp', ids = ids)
        comp_rel_save(array, folder2[4], 'step{}_'.format(i), image = False, arrow = True, gzip = True)
        #comp_save(array, folder2, 'step{}_'.format(i), arrow = True, gzip = True)
        #else:
            #raw_save(array, folder2, 'step{}_'.format(i), ids = ids, image = False, arrow=True)
    #print(final_shape)
    return final_shape

def make_compression_numpy(f2, f1, num_steps, folder_range):
    for i in folder_range:
        folder2 = [f2[j] + str(i) for j in range(len(f2))]
        folder1 = f1 + str(i)
        for f in folder2:
            try:
                shutil.rmtree(f)
            except OSError as e:
                pass
            os.mkdir(f)
        x, y = compression_convert(folder1, folder2, num_steps, dfile = '.pickle', input2 = [])
        # with open(os.path.join(folder2, 'x.pickle'), 'wb') as f:
        #     pickle.dump(x, f)
        # with open(os.path.join(folder2, 'y.pickle'), 'wb') as f:
        #     pickle.dump(y, f)

def make_compression_image(base_folder, f2, f1, num_steps):
    try:
        shutil.rmtree(base_folder)
    except OSError as e:
        pass
    os.mkdir(base_folder)
    for f in f2:
        try:
            shutil.rmtree(f)
        except OSError as e:
            pass
        os.mkdir(f)
    x, y = compression_convert(f1, f2, num_steps, dfile = '.npy', input2 = [])
    with open(os.path.join(base_folder, 'x.pickle'), 'wb') as f:
        pickle.dump(x, f)
    with open(os.path.join(base_folder, 'y.pickle'), 'wb') as f:
        pickle.dump(y, f)

def make_compression_relational(base_folder, f2, f1, num_steps):
    try:
        shutil.rmtree(base_folder)
    except OSError as e:
        pass
    os.mkdir(base_folder)
    for f in f2:
        try:
            shutil.rmtree(f)
        except OSError as e:
            pass
        os.mkdir(f)
    x, y = compression_convert(f1, f2, num_steps, dfile = '.pickle', input2 = [5])
    with open(os.path.join(base_folder, 'x.pickle'), 'wb') as f:
        pickle.dump(x, f)
    with open(os.path.join(base_folder, 'y.pickle'), 'wb') as f:
        pickle.dump(y, f)

if __name__ == '__main__':
    #print('hello 2')
    
    #folder1 = 'compression_tests_2/relational_pipeline'
    #folder2 = ['storage_10/numpy_raw', 'storage_10/numpy_pq', 'storage_10/numpy_gzip', 'storage_10/numpy_col', 'storage_10/numpy_dslog']
    folder2 = ['./storage_pipeline/storage_relational_compression/relational_raw', './storage_pipeline/storage_relational_compression/relational_pq', './storage_pipeline/storage_relational_compression/relational_gzip', './storage_pipeline/storage_relational_compression/relational_col', './storage_pipeline/storage_relational_compression/relational_dslog']
    base_folder = './storage_pipeline/storage_relational_compression/'

    folder2 = ['./storage_pipeline/storage_resnet_compression/resnet_raw', './storage_pipeline/storage_resnet_compression/resnet_pq', './storage_pipeline/storage_resnet_compression/resnet_gzip', './storage_pipeline/storage_resnet_compression/resnet_col', './storage_pipeline/storage_resnet_compression/resnet_dslog']
    base_folder = './storage_pipeline/storage_resnet_compression/'
    #folder1 = 'compression_tests_2/numpy_pipeline_10_'
    #folder_range = list(range(20))
    #folder_range = [13]
    #folder2 = ['./storage_pipeline/storage_image_compression/image_raw', './storage_pipeline/storage_image_compression/image_pq', './storage_pipeline/storage_image_compression/image_gzip', './storage_pipeline/storage_image_compression/image_col', './storage_pipeline/storage_image_compression/image_dslog']
    #make_compression_image( './storage_pipeline/storage_image_compression', folder2, 'compression_tests_2/image_pipeline', num_steps = 5) 
    #make_compression_numpy(folder2, folder1, 10, folder_range)
    #make_compression_relational(base_folder, folder2, 'compression_tests_3/resnet_pipeline', num_steps = 5)
    make_compression_relational(base_folder, folder2, 'compression_tests_2/resnet_pipeline', num_steps = 6)

