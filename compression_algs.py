

from cmath import isnan
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

def arr_save(array, path, name, ids = [1]):
    dfs = aux(array, ids)
    for id in dfs:
        dire = os.path.join(path, name + str(id) + '.npy')
        arr = dfs[id].values
        print(arr)
        np.save(dire, arr)
        
 
def raw_save(array, path, name, ids = [1], arrow = True):
    dfs = aux(array, ids)
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
    turbo_param = "-12"
    #tb_h = '-H'
    file_names = ['x1.csv', 'x2.csv', 'y1.csv', 'y2.csv']
    for temp in file_names:
        p1 = os.path.join(temp_path, temp)
        p2 = os.path.join(path, name + str(ids[0]))
        if not os.path.isdir(p2):
            os.mkdir(p2)
        p2 = os.path.join(p2, temp)
        #print(p2)
        #command = " ".join([turbo_dir, tb_h, turbo_param, p1, p2])
        command = " ".join([turbo_dir, turbo_param, p1, p2])
        print(command)
        os.system(command)

    if num_arrays == 2:
        temp_names = ['w1.csv', 'w2.csv', 'z1.csv', 'z2.csv']
        for i, temp in enumerate(temp_names):
            p1 = os.path.join(temp_path, temp)
            p2 = os.path.join(path, name + str(ids[0]))
            if not os.path.isdir(p2):
                os.mkdir(p2)
            p2 = os.path.join(p2, temp_names[i])
            #print(p2)
            command = " ".join([turbo_dir, turbo_param, p1, p2])
            print(command)
            os.system(command)

def convert_inverse_rel(prov):
    new_list = []
    for tup in prov:
        x1, x2  = tup[0]
        y1, y2  = tup[1]
        x, y = tup[2]
        max_i = -1
        temp = []
        if 'a' in x:
            type_x = 'a'
        elif '0' in x:
            type_x = '0'
        else:
            type_x = '1'
        if 'a' in y:
            type_y = 'a'
        elif '1' in y:
            type_y = '1'
        else:
            type_y = '0'
        
    
        max_i = len(x[type_x])

        for i in range(max_i):
            # if (type_x == '0' and type_y == '0') or (type_x == '1' and type_y == '1'):
            #     tups = get_rel_tups(tup[0], tup[1], x[type_x][i], y[type_y][i], type_x)
            #     new_list += tups
            #     continue

            # get xs
            out_x_tup = [None, None, None, None, None, None]
            out_y_tup = [None, None, None, None, None, None]
            if type_x == 'a' and type_y == 'a': #1
                x1_ = x[type_x][i][0]
                x2_ = x[type_x][i][1]
                y1_ = y[type_y][i][0]
                y2_ = y[type_y][i][1]
                out_x_tup[0] = x1
                out_x_tup[3] = x2
                out_y_tup[0] = y1
                out_y_tup[3] = y2
                tup = [x1_, x2_, y1_, y2_] + out_x_tup + out_y_tup 
                new_list.append(tup)
            elif type_x == 'a' and type_y == '1': #2
                x1_ = x[type_x][i][0]
                x2_ = x[type_x][i][1]
                out_x_tup[0] = x1
                out_x_tup[3] = x2
                y1_ = y1 + y[type_y][i][0]
                y2_ = y2 + y[type_y][i][1]
                out_y_tup[2] = y[type_y][i][0]
                out_y_tup[5] = y[type_y][i][1]
                tup = [x1_, x2_, y1_, y2_] + out_x_tup + out_y_tup 
                new_list.append(tup)
            elif type_x == '0' and type_y == 'a': #3
                y1_ = y[type_y][i][0]
                y2_ = y[type_y][i][1]
                out_y_tup[0] = y1
                out_y_tup[3] = y2
                x1_ = x1 + x[type_x][i][0]
                x2_ = x2 + x[type_x][i][1]
                out_x_tup[1] = x[type_x][i][0]
                out_x_tup[4] = x[type_x][i][1]
                tup = [x1_, x2_, y1_, y2_] + out_x_tup + out_y_tup 
                new_list.append(tup)
            elif type_x == '0' and type_y == '1': #4
                x1_ = x1 + x[type_x][i][0]
                x2_ = x2 + x[type_x][i][1]
                out_x_tup[1] = x[type_x][i][0]
                out_x_tup[4] = x[type_x][i][1]
                y1_ = y1 + y[type_y][i][0]
                y2_ = y2 + y[type_y][i][1]
                out_y_tup[2] = y[type_y][i][0]
                out_y_tup[5] = y[type_y][i][1]
                tup = [x1_, x2_, y1_, y2_] + out_x_tup + out_y_tup 
                new_list.append(tup)
            elif type_x == '1' and type_y == '0': #5
                y1_ = x1 + y[type_y][i][0]
                y2_ = x2 + y[type_y][i][1]
                x1_ = y1 + x[type_x][i][0]
                x2_ = y2 + x[type_x][i][1]
                out_x_tup[2] = y[type_y][i][0]
                out_x_tup[5] = y[type_y][i][1]
                out_y_tup[1] = x[type_x][i][0]
                out_y_tup[4] = x[type_x][i][1]
                tup = [x1_, x2_, y1_, y2_] + out_x_tup + out_y_tup 
                new_list.append(tup)
            elif type_x == 'a' and type_y == '0': #6
                x1_ = x[type_x][i][0]
                x2_ = x[type_x][i][1]
                out_x_tup[2] = y[type_y][i][0]
                out_x_tup[5] = y[type_y][i][1]
                y1_ = x1 + y[type_y][i][0]
                y2_ = x2 + y[type_y][i][1]
                out_y_tup[0] = y1
                out_y_tup[3] = y2
                tup = [x1_, x2_, y1_, y2_] + out_x_tup + out_y_tup 
                new_list.append(tup)
            elif type_x == '1' and type_y == 'a': #7
                y1_ = y[type_y][i][0]
                y2_ = y[type_y][i][1]
                out_y_tup[1] = x[type_x][i][0]
                out_y_tup[4] = x[type_x][i][1]
                x1_ = y1 + x[type_x][i][0]
                x2_ = y2 + x[type_x][i][1]
                out_x_tup[0] = x1
                out_x_tup[3] = x2
                tup = [x1_, x2_, y1_, y2_] + out_x_tup + out_y_tup 
                new_list.append(tup)
            elif type_x == '1' and type_y == '1': #8
                for j in range(y1, y2):
                    out_x_tup[0] = x1
                    out_x_tup[3] = x2
                    x1_ = j + x[type_x][i][0]
                    x2_ = j + x[type_x][i][1]
                    y1_ = j + y[type_y][i][0]
                    y2_ = j + y[type_y][i][1]
                    out_y_tup[1] = x[type_x][i][0]
                    out_y_tup[4] = x[type_x][i][1]
                    out_y_tup[2] = y[type_y][i][0]
                    out_y_tup[5] = y[type_y][i][1]
                    tup = [x1_, x2_, y1_, y2_] + out_x_tup + out_y_tup 
                    new_list.append(tup)
            elif type_x == '0' and type_y == '0': #9
                for j in range(x1, x2):
                    out_y_tup[0] = y1
                    out_y_tup[3] = y2
                    y1_ = j + y[type_y][i][0]
                    y2_ = j + y[type_y][i][1]
                    x1_ = j + x[type_x][i][0]
                    x2_ = j + x[type_x][i][1]
                    out_x_tup[1] = x[type_x][i][0]
                    out_x_tup[4] = x[type_x][i][1]
                    out_x_tup[2] = y[type_y][i][0]
                    out_x_tup[5] = y[type_y][i][1]
                    tup = [x1_, x2_, y1_, y2_] + out_x_tup + out_y_tup 
                    new_list.append(tup)            
    return new_list

offset = {'a': 0, '0': 1, '1':2}

def convert_rel(prov):
    vals = []
    for tup in prov:
        x1, x2  = tup[0]
        y1, y2  = tup[1]
        if x2 == x1:
            x2 = None
        if y2 == y1:
            y2 = None
        x, y = tup[2]
        tp_x = None
        tp_y = None
        for t in ['a', '0', '1']:
            if t in x and tp_x == None:
               tp_x = t
            if t in y and tp_y == None:
                tp_y = t
        for i in range(len(x[tp_x])):
            val = [None]*16
            val[0] = x1
            val[1] = x2
            val[2] = y1
            val[3] = y2
            val[4 + offset[tp_x]] = x[tp_x][i][0]
            if x[tp_x][i][1] != x[tp_x][i][0]:
                val[7 + offset[tp_x]] = x[tp_x][i][1]
            
            val[10 + offset[tp_y]] = y[tp_y][i][0]
            if y[tp_y][i][1] != y[tp_y][i][0]:
                val[13 + offset[tp_y]] = y[tp_y][i][1]
            vals.append(val)
    return vals
    
# (array, path, name,ids = [1], arrow = True)
def comp_rel_save(array, path, name, image = False, arrow = True, gzip = True):
    # import compression
    
    if image:
        provenance = {}
        provenance[1] = array_compression(array)
    else:
        provenance = compression(array, relative = True)
    #print(provenance)
    #print(provenance)
    for id in provenance:
        prov = provenance[id]
        vals = convert_rel(prov)
        vals_inverse = convert_inverse_rel(prov)
        df = pd.DataFrame(vals, columns= ["output_x1", "output_x2", "output_y1", "output_y2", \
                "input_x1_a", "input_x1_1", "input_x1_2", \
                "input_x2_a", "input_x2_1", "input_x2_2",  \
                "input_y1_a", "input_y1_1", "input_y1_2", \
                "input_y2_a", "input_y2_1", "input_y2_2"])
        pd.set_option('display.max_columns', None)
        #print(df)
        df_2 = pd.DataFrame(vals_inverse, columns= ["output_x1", "output_x2", "output_y1", "output_y2", \
                "input_x1_a", "input_x1_1", "input_x1_2", \
                "input_x2_a", "input_x2_1", "input_x2_2",  \
                "input_y1_a", "input_y1_1", "input_y1_2", \
                "input_y2_a", "input_y2_1", "input_y2_2"])
    
        if not arrow:
            dire = os.path.join(path, name + 'back' + str(id) + '.csv' )
            df.to_csv(dire)

            dire_2 = os.path.join(path, name + 'for' + str(id) + '.csv' )
            df_2.to_csv(dire_2)
        else:
            table = pa.Table.from_pandas(df, preserve_index=False)
            dire = os.path.join(path, name + 'back' + str(id) + '.parquet')
            if gzip:
                pq.write_table(table, dire, compression='gzip')
            else: 
                pq.write_table(table, dire)
            
            table_2 = pa.Table.from_pandas(df_2, preserve_index=False)
            dire_2 = os.path.join(path, name + 'for' + str(id) + '.parquet')
            if gzip:
               pq.write_table(table_2, dire_2, compression='gzip')
            else: 
               pq.write_table(table_2, dire_2)

def comp_save(array, path, name, arrow = True, gzip = True):
    provenance = compression(array, relative = False)
    
    for id in provenance:
        prov = provenance[id]
        vals = []
        for tup in prov:
            # insert input values
            x1, x2  = tup[0]
            y1, y2  = tup[1]
            if x1 == x2:
                x2 = None
            if y1 == y2:
                y2 = None
            for input in tup[2]:
                ix1, ix2 = input[0]
                iy1, iy2 = input[1]
                if ix2 == ix1:
                    ix2 = None
                if iy1 == iy2:
                    iy2 = None
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




array_size = [(100, 1), (1000, 1), (10000, 1), (100000, 1), (1000000, 1), (10000000, 1), (100000000, 1)]

if __name__ == '__main__':
    #for size in range(len(array_size)):
    # try:
    #     shutil.rmtree('./storage')
    # except OSError as e:
    #     pass
    # try:
    #     shutil.rmtree('./temp')
    # except OSError as e:
    #     pass
    # os.mkdir('./storage')
    # os.mkdir('./temp')        
    # with open ('./compression_tests_2/join_output.pickle', 'rb') as f:
    #     arr = pickle.load(f)
    #arr = test7(array_size[size])
    arr = test1()
    #arr = np.random.random((100, 100)).astype(tf.tracked_float)
    #tf.initialize(arr, 1)
    #arr = np.rot90(arr)
    #print(arr[0])
    # for i in range(0, 100):
    #     column_save(arr, './storage', 'step0_{}'.format(i), temp_path = './temp', ids = [1, 2])
        #print(i)
        #raw_save(arr[i], './storage', 'step0_{}'.format(i), ids = [1, 2], arrow = False)
    # start = time.time()
    
    arr_save(arr, './storage', 'step0_', ids = [1])
    #raw_save(arr, './storage', 'step0_', ids = [1], arrow = True)
    #column_save(arr, './storage', 'step0_', temp_path = './temp', ids = [1, 2])
    #gzip_save(arr, './storage', 'step0_', ids = [1, 2], arrow = True)
    #comp_rel_save(arr, './storage', 'step0_', image = False, arrow = True, gzip=False)
    #comp_save(arr, './storage', 'step0_', arrow = True, gzip=True)
    # end = time.time()
    #print('compression time')
    #print(array_size[size])
    #print(end - start)
    #print('compression size')
    #size = get_size(start_path = './storage')
    #print(size)        