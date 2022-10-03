import numpy as np
from query_compression import *
from query_dslog import *

def load_parquet(folder):
    tables = {}
    for name in os.listdir(folder):
        if name.endswith('.parquet'):
            dire = os.path.join(folder, name)
            nm = name.split('.')[0]
            tables[nm] = pq.read_table(dire)
    return tables
            # get folder name and last size
folder2 = 'storage/dslog_giz18'
            #folder2 = 'storage/turbo' + str(j)
            # with open(os.path.join(folder2, 'x.pickle'), 'rb') as f:
            #     x = pickle.load(f)
            # with open(os.path.join(folder2, 'y.pickle'), 'rb') as f:
            #     y = pickle.load(f)
            # get ranges and step names
#pranges = [((520, 520), (955, 965))]
# tnames = ['step3_for1']

# tables = load_parquet(folder2)
# table = tables['step3_for1']
# print(table)
tnames = []
pranges = [(0,1000), (0, 100)]
for i in range(5):
    tname = 'step{}_for1'.format(i)
    tnames.append(tname)
            #tnames.reverse()
            # get query results
#query_comp(pranges, folder2, tnames, merge = True, dtype = 'arrow')
            #result = query_one2one(pranges, folder2, tnames, backwards = False, dtype = 'arrow')
query_comp(pranges, folder2, tnames, merge = True, dtype = 'arrow')