from compression import compression
import numpy as np
import numpy.core.tracked_float as tf

def find_key(i, keys):
    if i in keys:
        return keys[i]
    else:
        return i

def find_key_tup(row, col, keys):
    row[0] = find_key(row[0], keys)
    row[1] = find_key(row[1], keys)
    col[0] = find_key(col[0], keys)
    col[1] = find_key(col[1], keys)

    return row, col

# prov_types = ('absabs', 'absrel', 'relabs', 'relrel')
def find_keys(com, keys):
    for arr in com:
        for (row, col, prov) in com[arr]:
            com[arr][0], com[arr][1] = find_key_tup(row, col, keys)
            for key in com[arr][2]:
                com[arr][2][key] = find_key_tup(com[arr][2][key], keys)

def compute_function(size):
    arr = np.random.random(size).astype(tf.tracked_float)
    tf.initialize(arr, 0)
    keys = {}
    # user defined here
    # bc i'm lazy and i'm the user -> not going to bother changing the argument
    # just change here
    # need to also define keys if needed
    #output = funct(*arrs)
    
    com = compression(output)
    print(com)
    if len(keys) > 0:
        com = find_keys(com, keys)
        print(com)