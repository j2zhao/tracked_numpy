
import pandas as pd
from relational_operations import *
import pickle
import os


def missing_filter(array, missing = '\\N'):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i,j].provenance = (0, i, j)
    new_array = []
    for i in range(array.shape[0]):
        skip = False
        for j in range(array.shape[1]):
            if array[i, j].value == missing:
                skip = True
                break
        if not skip:
            new_array.append(array[i, :].tolist())
    return np.array(new_array)

def combine_cols(array, col1 = 0, col2 = 0):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i,j].provenance = (0, i, j)
    
    new_array = np.zeros((array.shape[0], array.shape[1] + 1) , dtype= object)

    new_array[:, :-1] = array
    for i in range(array.shape[0]):
        new_array[i, -1] = TrackedObj('test', [(0, i, col1),(0, i, col2)])
    return new_array

def one_hot(array, column = 9):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i,j].provenance = (0, i, j)

    values = set()
    values_dict = {}
    index = 0
    for i in range(array.shape[0]):
        val = array[i, column].value
        if val not in values:
            values.add(val)
            values_dict[val] = index
            index +=1
    new_size = array.shape[1] - 1 + len(values)
    new_array = np.zeros((array.shape[0], new_size), dtype = object)
    new_array[:, :column] = array[:, :column]
    new_array[:, column:(array.shape[1] - 1)] = array[:, (column + 1):]
    for i in range(array.shape[0]): 
        for j in range(len(values)):
            x = j + array.shape[1] - 1
            new_array[i,x] = TrackedObj('onehot', [(0, i, column)])
    # for i in range(array.shape[0]): 
    #     val = array[i, column].value
    #     index = values_dict[val] + array.shape[1] - 1
    #     new_array[:, :column]
    return new_array
    
def one2one(array):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            array[i,j].provenance = (0, i, j)
    return array

if __name__ == '__main__':
    # pipeline
    folder = 'compression_tests_2/relational_pipeline'
    data_left = 'compression_tests_2/left_join_pandas.pickle'
    data_right = 'compression_tests_2/right_join_pandas.pickle'
    column1 = 'tconst'
    column2 = 'tconst'
    with open(data_left, 'rb') as f:
        df1 = pickle.load(f, encoding='latin1')
        df1 = df1.drop(labels='endYear', axis=1)
    with open(data_right, 'rb') as f:
        df2 = pickle.load(f, encoding='latin1')
    # Merge two datasets
    arr = join_prov(df1, df2, column1, column2, limit = 100000)
    p = os.path.join(folder, 'step1.pickle')
    with open(p, 'wb') as f:
        pickle.dump(arr, f)
    #Filter out missing data
    arr = missing_filter(arr)
    p = os.path.join(folder, 'step2.pickle')
    with open(p, 'wb') as f:
        pickle.dump(arr, f)
    #Create a new column that is the combination of two columns
    arr = combine_cols(arr)
    p = os.path.join(folder, 'step3.pickle')
    with open(p, 'wb') as f:
        pickle.dump(arr, f)
    # Use one hot encoding
    arr = one_hot(arr)
    p = os.path.join(folder, 'step4.pickle')
    with open(p, 'wb') as f:
        pickle.dump(arr, f)
    # range of one column
    arr = one2one(arr)
    p = os.path.join(folder, 'step5.pickle')
    with open(p, 'wb') as f:
        pickle.dump(arr, f)

    # Raise one column to exponents