import numpy as np
import os

def load_npy(folder):
    tables = {}
    for name in os.listdir(folder):
        if name.endswith('.npy'): 
            dire = os.path.join(folder, name)
            nm = name.split('.')[0]
            tables[nm] = np.load(dire)
    return tables

def query_array(pranges, folder, tnames, backwards = True, batch_size = 10):
    arrays = load_npy(folder)
    query_rows = []
    for prange in pranges:
        for i in range(prange[0][0], prange[0][1] + 1):
            for j in range(prange[1][0], prange[1][1] + 1):
                query_rows.append((int(i), int(j)))

    query_rows = np.asarray(query_rows)
    if len(query_rows.shape) == 1:
        query_rows = query_rows.reshape(-1, 1)
    for name in tnames:
        if backwards:
            cur_array = arrays[name][:, :2]
        else:
            cur_array = arrays[name][:, 2:]
        all_index_1 = []
        for i in range(0, query_rows.shape[0], batch_size):
            min_i = i
            max_i = min(i +batch_size, query_rows.shape[0])            
            query_rows_ = query_rows[min_i:max_i, :]
            #print(query_rows_)
            equal_rows = np.all(cur_array[:, np.newaxis] == query_rows_, axis=-1)
            index_1, index_2 = np.where(equal_rows)
            #print(index_1)
            all_index_1.append(index_1)
        all_index_1 = np.concatenate(all_index_1)
        if backwards:
            query_rows = arrays[name][all_index_1, 2:]
        else:
            query_rows = arrays[name][all_index_1, :2]
    return query_rows


if __name__ == '__main__':
     q = query_array([((0,10), (0,10))], 'storage', ['step0_1'], backwards = False)
     print(q)