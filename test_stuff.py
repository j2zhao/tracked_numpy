import numpy as np
from query_compression import *
from query_dslog import *
import time
import pickle
import pandas as pd
import duckdb



# Generating random data
query_rows = np.random.randint(low=0, high=100, size=(1000000, 2))
query_rows = pd.DataFrame(query_rows, columns=['col1', 'col2'])
query_rows_2 = np.random.randint(low=0, high=100, size=(1000000, 2))
query_rows_2 = pd.DataFrame(query_rows_2, columns=['col1', 'col2'])

# Connecting to DuckDB
con = duckdb.connect(database=':memory:')

# Registering dataframes as tables in DuckDB
con.register('query_rows', query_rows)
con.register('query_rows_2', query_rows_2)

# Preparing and executing the query
start = time.time()
query = '''
    SELECT *
    FROM query_rows
    JOIN query_rows_2
    ON query_rows.col1 = query_rows_2.col1
    AND query_rows.col2 = query_rows_2.col2
'''
result_df = con.execute(query).fetchdf()
end = time.time()

# Printing the time taken to execute the query
print(end - start)
# start = time.time()
# rows = []
# for row in query_rows.iterrows():
#     rows.append(row)
#     #for j in range(0, 1000):
#         #query_rows.append((int(i), int(j)))
# end = time.time()
# print(end-start)

# with open('./compression_tests_3/group_by_pandas_full.pickle', 'rb') as f:
#     table = pickle.load(f)

# print(table[1, 0].provenance)
# print('hello')

# def load_parquet(folder):
#     tables = {}
#     for name in os.listdir(folder):
#         if name.endswith('.parquet'):
#             dire = os.path.join(folder, name)
#             nm = name.split('.')[0]
#             tables[nm] = pq.read_table(dire)
#     return tables
#             # get folder name and last size
# folder2 = 'storage/dslog_giz18'
#             #folder2 = 'storage/turbo' + str(j)
#             # with open(os.path.join(folder2, 'x.pickle'), 'rb') as f:
#             #     x = pickle.load(f)
#             # with open(os.path.join(folder2, 'y.pickle'), 'rb') as f:
#             #     y = pickle.load(f)
#             # get ranges and step names
# #pranges = [((520, 520), (955, 965))]
# # tnames = ['step3_for1']

# # tables = load_parquet(folder2)
# # table = tables['step3_for1']
# # print(table)
# tnames = []
# pranges = [((0,1000), (0, 100))]
# for i in range(5):
#     tname = 'step{}_for1'.format(i)
#     tnames.append(tname)
#             #tnames.reverse()
#             # get query results
# #query_comp(pranges, folder2, tnames, merge = True, dtype = 'arrow')
#             #result = query_one2one(pranges, folder2, tnames, backwards = False, dtype = 'arrow')
# query_comp(pranges, folder2, tnames, merge = True, dtype = 'arrow')