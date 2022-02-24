'''
Generate full provenance of pairs:

((inputs), (outputs))

Note: matrix multiplication has two implementations with subzero defination?

'''


import numpy as np
import numpy.core.tracked_float as tf

def test1(arr_size = (10, 100000)):
    # basic test
    provenance = []
    for i in range(arr_size[0]):
        for j in range(arr_size[1]):
            prov = (((1, i, j)), ((2, i, j)))
            provenance.append(prov)
    return provenance

def test2(arr = (10, 100000), arr2 = (10, 100000)):
    # test 2 arrays
    provenance = []
    for i in range(arr[0]):
        for j in range(arr[1]):
            prov = (((1, i, j)), (2, i, j), (3, i, j))
            provenance.append(prov)
    return provenance

def test3(arr = (10, 100000)):
    # test reduction
    provenance = []
    for i in range(arr[0]):
        out = (1, i, 1)
        prov = []
        for j in range(arr[1]):
            prov.append((2, j, 1))
        pr = ((out), tuple(prov))
        provenance.append(pr)
    return provenance

def test4(arr = (10, 100000)):
    #tests duplication (2x2)
    prov = []
    l = arr[0]
    w = arr[1]
    for i in range(arr[0]):
        for j in range(arr[1]):
            pr = (((1, i, j)), ((2, i, j), (2, i + l, j), (2, i, j + w), (2, i + l, j + w)))
            prov.append(pr)
    return prov

def test5(arr = (10, 100000)):
    #tests duplication (3 x 1 )
    prov = []
    l = arr[0]
    w = arr[1]
    for i in range(arr[0]):
        for j in range(arr[1]):
            pr = (((1, i, j)), ((2, i, j), (2, i + l, j), (2, i + l*2, j)))
            prov.append(pr)
    return prov

def test6(arr = (1000, 1000), arr2 = (1000, 1000)):
    #test matmul
    prov = []
    for k in range(arr[0]):
        out = []
        for i in range(arr[1]):
            out.append((2, k, i))
        input = []
        for j in range(arr2[1]):
            input.append((1, k, j))
        prov.append(tuple(input), tuple(out))
    for k in range(arr2[1]):
        out = []
        for i in range(arr2[0]):
            out.append((2, i, k))
        input = []
        for j in range(arr2[1]):
            input.append((1, j, k))
        prov.append(tuple(input), tuple(out))
    return prov

def test7(arr = (10000000, 1), arr2 = (10000000, 1)):
    # test vector*vector
    prov = []
    pr = []
    for i in range(arr[0]):
        pr.append((2, i, 1))
        pr.append((3, i, 1))
    prov2 = ((1, 1, 1), tuple(pr))
    prov.append(prov2)
    return prov

def test8(arr = (1000, 1000), arr2 = (1000, 1)):
    # test matrix*vector
    prov = []
    for k in range(arr[0]):
        out = []
        for i in range(arr[1]):
            out.append((2, k, i))
        prov.append(((1, k, 1)), tuple(out))
    
    out = []
    for i in range(arr2[0]):
        out.append((2, i, 1))
    input = []
    for i in range(arr[0]):
        input.append((1, i, 1))
    prov.append(tuple(input), tuple(out))
    # for j in range(arr[0]):
    #     pr = []
    #     for i in range(arr[1]):
    #         pr.append((2, j, i))
    #         pr.append((3, i, 1))
    #     prov2 = ((1, j, 1), tuple(pr))
    #     prov.append(prov2)

    return prov

def test9(prov_arr, arr = (1000000, 1)):
    # tests filters
    prov = []
    for i in range(arr[0]):
        if prov_arr[i].n != 0:
            pr = ((0, i, 1), (2, i, 1))
            prov.append(pr)
    return prov    

def test10(prov_arr, arr = (1000000, 1)):
    prov = []
    for i in range(arr[0]):
        if prov_arr[i].n != 0:
            pr = ((0, i, 1), (2, i, 1))
            prov.append(pr)
    return prov 

def test11(prov_arr):
    # return first array not out array
    prov0 = []
    prov1 = []
    prov2 = []
    prov3 = []
    for i in range(prov_arr.shape[0]):
        if prov_arr[i].n <= 0.25:
            prov0.append((1, i, 1))
        elif prov_arr[i].n <= 0.5:
            prov1.append((1, i, 1))
        elif prov_arr[i].n <= 0.75:
            prov2.append((1, i, 1))
        else:
            prov3.append((1, i, 1))

    prov = []
    pr = ((0, 0, 1), tuple(prov0))
    prov.append(pr)
    prov = []
    pr = ((0, 1, 1), tuple(prov1))
    prov.append(pr)
    prov = []
    pr = ((0, 2, 1), tuple(prov2))
    prov.append(pr)
    prov = []
    pr = ((0, 3, 1), tuple(prov3))
    prov.append(pr)
    return prov 

def test12(prov_arr):
    # return first array not out array
    prov0 = []
    prov1 = []
    prov2 = []
    prov3 = []
    for i in range(prov_arr.shape[0]):
        if prov_arr[i].n <= 0.25:
            prov0.append((1, i, 1))
        elif prov_arr[i].n <= 0.5:
            prov1.append((1, i, 1))
        elif prov_arr[i].n <= 0.75:
            prov2.append((1, i, 1))
        else:
            prov3.append((1, i, 1))

    prov = []
    pr = ((0, 0, 1), tuple(prov0))
    prov.append(pr)
    prov = []
    pr = ((0, 1, 1), tuple(prov1))
    prov.append(pr)
    prov = []
    pr = ((0, 2, 1), tuple(prov2))
    prov.append(pr)
    prov = []
    pr = ((0, 3, 1), tuple(prov3))
    prov.append(pr)
    return prov 


def test13(prov_arr):
    prov = []
    for i in range(prov_arr.shape[0]):
        if prov_arr[i].n != 0:
            pr = ((0, i, 1), (2, i, 1))
            prov.append(pr)
    return prov 