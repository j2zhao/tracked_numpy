
import numpy as np
import os
import numpy.core.tracked_float as tf
import json
import sys
import random

def convert_functions(func):
    f = []
    f.append(getattr(np, func[0]))
    kwargs = {}
    for key, val in func[1]:
        if func[0] == 'partition' and key == 'kth':
            kwargs[key] = random.randrange(0, 1000)
        elif func[0] == 'delete' and key == 'obj':
            kwargs[key] = random.randrange(0, 1000)
        elif func[0] == 'insert' and key == 'obj':
            kwargs[key] = random.randrange(0, 1000)
        elif func[0] == 'insert' and key == 'values':
            kwargs[key] = np.zeros(()).astype(tf.tracked_float)
        else:
            index = random.randrange(len(val))
            kwargs[key] = val[index]

def run_function(array, f, kwargs):
    if len(array.shape) == 0:
        array  = np.reshape(array, (1,1))
        tf.initialize(array, i)
    elif len(array.shape) == 1:
        array  = np.reshape(array, (-1,1))
        tf.initialize(array, i)
    else:
        tf.initialize(array, i)
    output = f(array, **kwargs)
    return output

if __name__ == '__main__':
    folder = ''
    func_list = ''
    size = int(sys.argv[1])
    # read function list
    with open(func_list, 'r') as f:
        func = f.readlines()
        func = [json.loads(row) for row in func]
    # choose functions
    n = len(func)
    indices = [random.randrange(0, n) for i in range(size)]
    func2 = []

    # run functions
    array = np.zeros((1000, 1000))
    for i, f in enumerate(func2):
        array, prov = run_function(array, f[0], f[1])
        dire = os.path.join(folder, 'step{}.npy'.format())
        np.save(dire, prov)