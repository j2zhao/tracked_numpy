
import numpy as np
import os
import numpy.core.tracked_float as tf
import json
import sys
import random

def convert_functions(func):
    print(func)
    kwargs = {}
    if len(func) < 2:
        return (getattr(np, func[0]), {})
    for key, val in func[1].items():
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
    print(kwargs)
    return (getattr(np, func[0]), kwargs)

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
    folder = 'compression_tests_2/numpy_pipeline'
    func_list = 'compression_tests_2/single_functions.txt'
    size = int(sys.argv[1])
    # read function list
    func = []
    with open(func_list, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            func.append(json.loads(lines[i]))
    # choose functions
    n = len(func)
    indices = [random.randrange(0, n) for i in range(size)]
    #indices = list(range(n))
    func2 = []
    for i in indices:
        func2.append(convert_functions(func[i]))
    # run functions
    array = np.random.rand(1000, 1000)
    for i, f in enumerate(func2):
        array = run_function(array, f[0], f[1])
        dire = os.path.join(folder, 'step{}.npy'.format(i))
        np.save(dire, array)