#from errno import ESHLIBVERS
import re
import numpy as np
import time
    

#to fix
def compress_output(prov_arr, value = 1):
    '''
    we compress by specified id -> requires id field
    '''
    compressed = []
    prev_compressed_col = None
    for row in range(prov_arr.shape[0]):
        temp_start = -1
        last_value = -1
        prov1 = -1 
        compressed_col = []
        for col in range(prov_arr.shape[1]):
            # if previous interval is empty
            if prov1 == -1:
                if prov_arr[row, col] == value:
                    temp_start = col
                    last_value = col
                    prov1 = ({'0': [(0, 0)]}, {'1': [(0, 0)]})
                # need to check alternative??
            else:
                # check for provenance and match
                if prov_arr[row, col] == value:
                    prov2 = prov1
                else:
                    prov2 = -1
                # compression step
                if prov2 == -1:
                    compressed_col.append(((row, row), (temp_start, last_value), prov1))
                    temp_start = -1
                    last_value = -1
                    prov1 = -1
                else:
                    last_value = col        
        if prov1 != -1:
            compressed_col.append(((row, row), (temp_start, last_value), prov1))
        
        if prev_compressed_col == None:
            prev_compressed_col = compressed_col
        else:
            prev_index = 0
            new_compressed_col = []
            for interval in compressed_col:
                while True:
                    if prev_index >= len(prev_compressed_col):
                        new_compressed_col.append(interval)
                        break
                    prev = prev_compressed_col[prev_index]
                    if prev[1][0] < interval[1][0]:
                        compressed.append(prev)
                        prev_index +=1
                    elif prev[1][0] == interval[1][0]:
                        prov = prev[2]
                        if prev[1][1] == interval[1][1] and len(prov) != 0:
                            new_compressed_col.append(((prev[0][0], interval[0][1]), interval[1], prov))
                            prev_index +=1
                        else:
                            compressed.append(prev)
                            prev_index +=1
                            new_compressed_col.append(interval)
                        break
                    else:
                        new_compressed_col.append(interval)
                        break
            if len(prev_compressed_col) > prev_index:
                for i in range(prev_index, len(prev_compressed_col)):
                    compressed.append(prev_compressed_col[i])
            prev_compressed_col = new_compressed_col

    if prev_compressed_col != None:
        compressed += prev_compressed_col
    #print(compressed)
    return compressed


def array_compression(prov_arr):
    '''
    '''
    # change this format based on whatever
    # need to change to -1 for compression
    # merge inputs
    if len(prov_arr.shape) == 0:
        prov_arr = np.reshape(prov_arr, (1, 1))
    if len(prov_arr.shape) == 1:
        prov_arr = np.reshape(prov_arr, (prov_arr.shape[0], 1))
    output = compress_output(prov_arr)
    return output

    

if __name__ == '__main__':

    # base = './logs'
    # prov = np.load('logs/1632433790.421791.npy', allow_pickle=True)
    prov = np.zeros((100, 100))
    prov[0:50, 0:50] = 1
    compressed = array_compression(prov)
    print(compressed)
