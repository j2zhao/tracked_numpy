

import numpy as np
import time
import os
import gzip
import json
from compression import compression
from compression_examples import *
import col_compression 
import time
import shutil
import pickle
import pyarrow as pa
import pandas as pd
from compression_array import array_compression
import pyarrow.parquet as pq
import shutil


def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def aux(array, ids = [1]):
    if len(array.shape) == 0:
        array = np.reshape(array, (1, 1))
    if len(array.shape) == 1:
        array = np.reshape(array, (array.shape[0], 1))
    provs = {}
    for id in ids:
        provs[id] = []
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for p in array[i, j].provenance:
                id, x, y = p
                provs[id].append([i, j, x, y])
    results = {}
    for id in ids:
        results[id] = pd.DataFrame(provs[id], columns=['output_x', 'output_y', 'input_x', 'input_y'])
    return results
 
def raw_save(array, path, name, ids = [1], arrow = True):
    dfs = aux(array, ids)
    print('hello')
    for id in dfs:
        if not arrow:
            dire = os.path.join(path, name + str(id) + '.csv')
            dfs[id].to_csv(dire)
        else:
            table = pa.Table.from_pandas(dfs[id], preserve_index=False)
            dire = os.path.join(path, name + str(id) + '.parquet')
            pq.write_table(table, dire)

def gzip_save(array, path, name, ids = [1], arrow = True):
    dfs = aux(array, ids)
    for id in dfs:
        if not arrow:
            dire = os.path.join(path, name + str(id) + '.csv')
            df = dfs[id].to_csv()
            com = gzip.compress(df)
            with open(dire, 'wb') as f:
                f.write(com)
        else:
            table = pa.Table.from_pandas(dfs[id], preserve_index=False)
            dire = os.path.join(path, name + str(id) + '.parquet')
            pq.write_table(table, dire, compression='gzip')


def inverted_list(array, path, name, ids = [1], batch_size = 10000, arrow = True):
    """
    thinking about it -> the fastest way might be list and pickle 
    """
    if len(array.shape) == 0:
        array = np.reshape(array, (1, 1))
    if len(array.shape) == 1:
        array = np.reshape(array, (array.shape[0], 1))

    temp_batches = {}
    temp_indices = {}
    temp_array = {}
    for id in ids:
        temp_batches[id] = np.empty((batch_size,))
        temp_indices[id] = 0
        temp_array[id] = 1
    
    provs = {}
    for id in ids:
        provs[id] = []

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if len(array[i, j].provenance) == 1:
                id, x, y = array[i, j].provenance[0]
                provs[id].append([i, j, x, y])
            else:
                temps = {}
                for p in array[i, j].provenance:
                    id, x, y = p
                    if id not in temps:
                        temps[id] = []
                    temps[id].append((x, y))
                
                for id in temps:
                    index = temp_indices[id]
                    if batch_size - index <= len(temps[id])*2:
                        dire = os.path.join(path, name +  str(id) + '_' + str(temp_array[id]) + '.npy')
                        np.save(dire, temp_batches[id])
                        temp_batches[id] = np.empty((batch_size))
                        index = 0
                        temp_array[id] += 1

                    provs[id].append((i, j, -index, -temp_array[id]))
                    
                    for x, y in temps[id]:
                        temp_batches[id][index] = x
                        temp_batches[id][index + 1] = y
                        index += 2
                    
                    temp_batches[id][index] = -1
                    temp_indices[id] = index + 1
    
    for id in temp_batches:
        dire = os.path.join(path, name +  str(id) + '_' + str(temp_array[id]) + '.npy')
        np.save(dire, temp_batches[id])

    results = {}
    for id in ids:
        results[id] = pd.DataFrame(provs[id], columns=['output_x', 'output_y', 'input_x', 'input_y'])
        if not arrow:
            dire = os.path.join(path, name + str(id) + '.csv')
            results[id].to_csv(dire)
        else: 
            table = pa.Table.from_pandas(results[id], preserve_index=False)
            dire = os.path.join(path, name + str(id) + '.parquet')
            pq.write_table(table, dire)

# (array, path, name,ids = [1], arrow = True)
def column_save(array, path, name, temp_path = './temp', ids = [1]):
    num_arrays = len(ids)
    if len(array.shape) == 0:
        array = np.reshape(array, (1, 1))
    if len(array.shape) == 1:
        array = np.reshape(array, (array.shape[0], 1))
    if num_arrays == 1:
        col_compression.to_column_1(array, temp_path, zeros = True)
    else:
        col_compression.to_column_2(array, temp_path, ids, zeros = True)
    turbo_dir = "./turbo/Turbo-Range-Coder/turborc"
    turbo_param = "-20"
    tb_h = '-FH'
    file_names = ['x1.npy', 'x2.npy', 'y1.npy', 'y2.npy']
    for temp in file_names:
        p1 = os.path.join(temp_path, temp)
        p2 = os.path.join(path, name + str(ids[0]))
        if not os.path.isdir(p2):
            os.mkdir(p2)
        p2 = os.path.join(p2, temp)
        print(p2)
        command = " ".join([turbo_dir, tb_h, turbo_param, p1, p2])
        print(command)
        os.system(command)

    if num_arrays == 2:
        temp_names = ['w1.npy', 'w2.npy', 'z1.npy', 'z2.npy']
        for i, temp in enumerate(temp_names):
            p1 = os.path.join(temp_path, temp)
            p2 = os.path.join(path, name + str(ids[0]))
            if not os.path.isdir(p2):
                os.mkdir(p2)
            p2 = os.path.join(p2, temp_names[i])
            command = " ".join([turbo_dir, tb_h, turbo_param, p1, p2])
            os.system(command)
    
# (array, path, name,ids = [1], arrow = True)
def comp_rel_save(array, path, name, image = False, arrow = True, gzip = True):
    # import compression
    if image:
        provenance = {}
        provenance[1] = array_compression(array)
    else:
        provenance = compression(array, relative = True)
    print(provenance)
    for id in provenance:
        prov = provenance[id]
        vals = []
        i = 0
        for tup in prov:
            i +=1
            # insert input values
            x1, x2  = tup[0]
            y1, y2  = tup[1]
            x, y = tup[2]
            i = 0
            max_i = -1
            while True:
                val = [x1, x2, y1, y2]
                for t in ['a', '0', '1']:
                    if t in x:
                        if max_i == -1:
                            max_i = len(x[t])
                        val.append(x[t][i][0])
                    else:
                        val.append(None)

                for t in ['a', '0', '1']:
                    if t in x:
                        val.append(x[t][i][1])
                    else:
                        val.append(None)

                for t in ['a', '0', '1']:
                    if t in y:
                        val.append(y[t][i][0])
                    else:
                        val.append(None)
                for t in ['a', '0', '1']:
                    if t in y:
                        val.append(y[t][i][1])
                    else:
                        val.append(None)

                vals.append(val)
                if i >= max_i - 1:
                    break
                i += 1
        df = pd.DataFrame(vals, columns= ["output_x1", "output_x2", "output_y1", "output_y2", \
                "input_x1_a", "input_x1_1", "input_x1_2", \
                "input_x2_a", "input_x2_1", "input_x2_2",  \
                "input_y1_a", "input_y1_1", "input_y1_2", \
                "input_y2_a", "input_y2_1", "input_y2_2"])
        if not arrow:
            dire = os.path.join(path, name + str(id) + '.csv' )
            df.to_csv(dire)
        else:
            table = pa.Table.from_pandas(df, preserve_index=False)
            dire = os.path.join(path, name + str(id) + '.parquet')
            if gzip:
                pq.write_table(table, dire, compression='gzip')
            else: 
                pq.write_table(table, dire)

def comp_save(array, path, name, arrow = True, gzip = True):
    provenance = compression(array, relative = False)
    for id in provenance:
        prov = provenance[id]
        vals = []
        for tup in prov:
            # insert input values
            x1, x2  = tup[0]
            y1, y2  = tup[1]
            for input in tup[2]:
                ix1, ix2 = input[0]
                iy1, iy2 = input[1]
                val = [x1, x2, y1, y2, ix1, ix2, iy1, iy2]
                vals.append(val)
        df = pd.DataFrame(vals, columns= ["output_x1", "output_x2", "output_y1", "output_y2", \
                 "input_x1", "input_x2", \
                "input_y1", "input_y2"])
        if not arrow:
            dire = os.path.join(path, name + str(id) + '.csv' )
            df.to_csv(dire)
        else:
            table = pa.Table.from_pandas(df, preserve_index=False)
            dire = os.path.join(path, name + str(id) + '.parquet')
            if gzip:
                pq.write_table(table, dire, compression='gzip')
            else: 
                pq.write_table(table, dire)
import matplotlib.pyplot as plt


array_size = [(100, 1), (1000, 1), (10000, 1), (100000, 1), (1000000, 1), (10000000, 1), (100000000, 1)]

if __name__ == '__main__':
    for size in array_size:
        try:
            shutil.rmtree('./storage')
        except OSError as e:
            pass
        try:
            shutil.rmtree('./temp')
        except OSError as e:
            pass
        os.mkdir('./storage')
        os.mkdir('./temp')        
        # with open ('./compression_tests_2/join_output.pickle', 'rb') as f:
        #     arr = pickle.load(f)
        arr = test7(arr_shape=size)
        #imgplot = plt.imshow(arr)
        #plt.show()
        # print('done')
        
        # print('hello')
        #column_save(arr, './storage', 'step0_', temp_path = './temp', ids = [1])
        #gzip_save(arr, './storage', 'step0_', ids = [1], arrow = True)
        # for i in range(900, 1000):
        #     print(i)
        #     column_save(arr[i], './storage', 'step0_{}'.format(i), temp_path = './temp', ids = [1,2])
        #     # raw_save(arr[i], './storage', 'step0_{}'.format(i), ids = [1, 2], arrow = False)
        # print('generated array')
        # print(arr.shape)
        # with open( './compression_tests_2/join_output.pickle', 'wb') as f:
        #     pickle.dump(arr, f)
        # with open ('./temp/join_output_2.pickle', 'rb') as f:
        #     arr = pickle.load(f)
        # for i in range(100, 200):
        #     print(i)
            # raw_save(arr[i], './storage', 'step0_{}'.format(i), ids = [1, 2], arrow = False)
        start = time.time()
        #column_save(arr, './storage', 'step0_', temp_path = './temp', ids = [1])
        raw_save(arr, './storage', 'step0_', ids = [1], arrow = True)
        #comp_rel_save(arr, './storage', 'step0_', arrow = True, gzip=True)
        end = time.time()
        print('compression time')
        print(size)
        print(end - start)
        # print('compression size')
        # size = get_size(start_path = './storage')
        # print(size)