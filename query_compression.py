
import duckdb
import os
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import time


def load_parquet(folder):
    tables = {}
    for name in os.listdir(folder):
        if name.endswith('.parquet'):
            dire = os.path.join(folder, name)
            nm = name.split('.')[0]
            tables[nm] = pq.read_table(dire)
    return tables

def load_csv(folder):
    tables = {}
    for name in os.listdir(folder):
        if name.endswith('.csv'): 
            dire = os.path.join(folder, name)
            nm = name.split('.')[0]
            tables[nm] = pd.read_csv(dire)
    return tables

def load_turbo(folder):
    turbo_dir = "./turbo/Turbo-Range-Coder/turborc"
    turbo_param = "-d"
    tables = {}
    for f1 in os.listdir(folder):
        f2 = os.path.join(folder, f1)
        if os.path.isdir(f2):
            file_names = ["x1.csv", "x2.csv", "y1.csv", "y2.csv"]
            db_names = ['output_x', 'output_y', 'input_x', 'input_y']
            array_dict = {}
            for i, file in enumerate(file_names):
                p1 = os.path.join(f2, file + '.rc')
                p2 = os.path.join(f2, file)
                command = " ".join([turbo_dir, turbo_param, p1, p2])
                os.system(command)
                #array_dict[db_names[i]] = np.load(p2, allow_pickle=True)
                #print(p2)
                x = np.genfromtxt(p2, delimiter=',')
                x = np.reshape(x, (-1))
                array_dict[db_names[i]] = x
                #print(array_dict[db_names[i]].shape)
            table = pd.DataFrame(array_dict)
            tables[f1] = table
    return tables


def load_temp():
    tables = {}
    file_names = ["x1.npy", "x2.npy", "y1.npy", "y2.npy"]
    db_names = ['input_x', 'input_y', 'output_x', 'output_y']
    array_dict = {}
    for i, file in enumerate(file_names):
        file = os.path.join('temp', file)
        array_dict[db_names[i]] = np.load(file, allow_pickle=True)
    table = pd.DataFrame(array_dict)
    tables['step0_1'] = table
    return tables

def query_one2one(pranges, folder, tnames, backwards = True, dtype = 'arrow'):
    """
    type: 'arrow' or 'turbo' or 'csv' -> right now only support  arrow and turbo

    """
    con = duckdb.connect(database=':memory:')
    if dtype == 'arrow':
        tables = load_parquet(folder)
    elif dtype == 'turbo':
        tables = load_turbo(folder)
    elif dtype == 'csv':
        tables = load_csv(folder)
    elif dtype == 'temp':
        tables = load_temp()
    else:
        raise ValueError('dtype argument not supported')

    query_rows = []
    #print(pranges)
    for prange in pranges:
        for i in range(prange[0][0], prange[0][1] + 1):
            for j in range(prange[1][0], prange[1][1] + 1):
                query_rows.append((int(i), int(j)))
    query_rows = pd.DataFrame(query, columns=['output_x', 'output_y'])
    start = time.time()
    total = 0
    for name in tnames:
        arrow_table = tables[name]
        new_query_rows = set()
        con.register('query_rows_table', query_rows)
        con.register('query_rows_table', arrow_table)
        # for row in query_rows:
        #     row = (int(row[0]), int(row[1]))
            # print(con.fetchall())
        if backwards:
            #query = 'SELECT input_x, input_y FROM arrow_table WHERE (output_x, output_y) IN ' + str(tuple(query_rows))
            con.execute(query)
            #con.execute('SELECT input_x, input_y FROM arrow_table WHERE output_x = ? AND output_y = ?', row)
            sql_results = con.fetchdf()
            for _, row in sql_results.iterrows():
                new_query_rows.add((row['input_x'], row['input_y']))
        else:
            start = time.time()
            #query = 'SELECT output_x, output_y FROM arrow_table WHERE (input_x, input_y) IN ' + str(tuple(query_rows))
            query = 'SELECT output_x, output_y FROM arrow_table WHERE arrow_table.input_x = query_rows_table.output_x, AND arrow_table.input_y = query_rows_table.output_y'
            #query = 'SELECT * FROM arrow_table'
            con.execute(query)
            #con.execute('SELECT output_x, output_y FROM arrow_table WHERE input_x = ? AND input_y = ?', row)
            sql_results = con.fetchdf()
            end = time.time()
            # for _, row in sql_results.iterrows():
            #     new_query_rows.add((row['output_x'], row['output_y']))
        #query_rows = new_query_rows
        query_rows = sql_results
        print(query_rows)
        if len(query_rows) == 0:
            return query_rows
    print(total)
    #return(start-end)
    return query_rows

# def query_invertedlist(pranges, folder, tnames, dtype = 'arrow'):
#     con = duckdb.connect(database=':memory:')
#     if dtype == 'arrow':
#         tables = load_parquet(folder)

#     query_rows = []
#     for prange in pranges:
#         for i in range(prange[0][0], prange[0][1] + 1):
#             for j in range(prange[1][0], prange[1][1] + 1):
#                 query_rows.append([i, j])
#     for name in tnames:
#         arrow_table = tables[name]
#         # print(con.fetchall())
#         new_query_rows = []
#         for row in query_rows:
#             arrow_table = tables[name]
#             sql_results = con.execute('SELECT input_x, input_y FROM arrow_table WHERE output_x = ? AND output_y = ?', row).fetchdf()
#             indices = {}
#             for _, row in sql_results.iterrows():
#                 if int(row['input_x']) >= 0 and int(row['input_y']) >= 0:
#                     new_query_rows.append((row['input_x'], row['input_y']))
#                 else:
#                     index = abs(int(row['input_x']))
#                     array_id = abs(int(row['input_y']))
#                     if array_id not in indices:
#                         indices[array_id] = []
#                     indices[array_id].append(index)
#         for i in indices:
#             dire = os.path.join(folder, name + '_' + str(i) + '.npy')
#             arr = np.load(dire).tolist()
#             for j in indices[i]:
#                 arr_i = j
#                 while True:
#                     if arr[arr_i] == -1:
#                         break
#                     new_query_rows.append([int(arr[arr_i]), int(arr[arr_i + 1])])
#                     arr_i += 2
#         query_rows = new_query_rows
#     return query_rows

if __name__ == '__main__':
    q = query_one2one([((0,0), (0,0))], 'storage', ['step0_1'], backwards = True, dtype = 'turbo')
    print(q)