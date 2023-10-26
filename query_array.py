import numpy as np
import os

def load_npy(folder):
    tables = {}
    for name in os.listdir(folder):
        if name.endswith('.csv'): 
            dire = os.path.join(folder, name)
            nm = name.split('.')[0]
            tables[nm] = np.load(dire)
    return tables

def query_one2one(pranges, folder, tnames, backwards = True, batch_size = 1000, dtype = 'arrow'):
    arrays = load_npy(folder)
    query_rows = []
    #print(pranges)
    for prange in pranges:
        for i in range(prange[0][0], prange[0][1] + 1):
            for j in range(prange[1][0], prange[1][1] + 1):
                query_rows.append((int(i), int(j)))
    
    query_rows = np.asarray(prange)
    for name in tnames:
        if backwards:
            cur_array = arrays[name][:2]
        else:
            cur_array = arrays[name][2:]
        
        all_index_1 = []
        for i in range(0, query_rows.shape[0], batch_size):
            max_i = min((i +1)*batch_size, query_rows.shape[0])
            min_i = i*batch_size
            query_rows = query_rows[min_i:max_i, :]
            equal_rows = np.all(cur_array[:, np.newaxis] == query_rows, axis=-1)
            index_1, index_2 = np.where(equal_rows)
            all_index_1.append(index_1)
        
        all_index_1 = np.concatenate(all_index_1)
        if backwards:
            query_rows = arrays[name][2:, all_index_1]
        else:
            query_rows = arrays[name][2:, all_index_1]
    return query_rows