

import duckdb
import os
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from query_compression import load_parquet
import time

def sort_(prov):
    (ox1, ox2), (oy1, oy2) = prov
    s = str(ox1) + str(ox2) + str(oy1) + str(oy2)
    return int(s)

def merge_ranges(pranges):
    """pranges [(x1, x2), (y1, y2)"""
    plist = list(set(pranges))
    plist.sort(key=sort_)
    cur_x1 = -1
    cur_x2 = -1
    temp_start = -1
    last_value = -1
    temp_list = {}
    for prange in plist:
        if cur_x1 == -1:
            cur_x1 = prange[0][0]
            cur_x2 = prange[0][1]
            temp_start = prange[1][0]
            last_value = prange[1][1]
        elif cur_x1 != prange[0][0] and cur_x2 != prange[0][1]:
            if (temp_start, last_value) not in temp_list:
                temp_list[(temp_start, last_value)] = []
            temp_list[(temp_start, last_value)].append((cur_x1, cur_x2))
            cur_x1 = prange[0][0]
            cur_x2 = prange[0][1]
            temp_start = prange[1][0]
            last_value = prange[1][1]
        elif last_value >= prange[1][0] - 1:
            last_value = max(prange[0][1], last_value)
        else:
            if (temp_start, last_value) not in temp_list:
                temp_list[(temp_start, last_value)] = []
            temp_list[(temp_start, last_value)].append((cur_x1, cur_x2))
            temp_start = prange[1][0]
            last_value = prange[1][1]
    if temp_start != -1 and (temp_start, last_value) not in temp_list:
        temp_list[(temp_start, last_value)] = []
    temp_list[(temp_start, last_value)].append((cur_x1, cur_x2))
    compressed = []
    for col in temp_list:
        temp_start = -1
        last_value = -1
        for x in temp_list[col]:
            if temp_start == -1:
                temp_start = x[0]
                last_value = x[1]
            elif last_value >= x[0] - 1:
                last_value = max(last_value, x[1])
            else:
                compressed.append(((temp_start, last_value), col))
                temp_start = x[0]
                last_value = x[1]
        compressed.append(((temp_start, last_value), col))
    return compressed

def input_output(prange, results):
    x1 = prange[0][0]
    x2 = prange[0][1]
    y1 = prange[1][0]
    y2 = prange[1][1]

    # 0: a, 1: 1, 2: 2
    oranges = []
    for row in results.itertuples():
        # get intersection with row range
        sql_x1 = int(row[1])
        sql_x2 = int(row[2])
        sql_y1 = int(row[3])
        sql_y2 = int(row[4])

        ix1 = max(x1, sql_x1)
        ix2 = min(x2, sql_x2)
        iy1 = max(y1, sql_y1)
        iy2 = min(y2, sql_y2)
        # for each output find appropriate range

        for i in range(3):
            j = i + 5
            k = j + 3
            if not np.isnan(row[j]):
                if i == 0:
                    ox1 = int(row[j])
                    ox2 = int(row[k])
                elif i == 1:
                    ox1 = int(row[j]) + ix1
                    ox2 = int(row[k]) + ix2
                else:
                    ox1 = int(row[j]) + iy1
                    ox2 = int(row[k]) + iy2
                break

        for i in range(3):
            j = i + 11
            k = j + 3
            if not np.isnan(row[j]):
                if i == 0:
                    oy1 = int(row[j])
                    oy2 = int(row[k])
                elif i == 1:
                    oy1 = int(row[j]) + ix1
                    oy2 = int(row[k]) + ix2
                else:
                    oy1 = int(row[j]) + iy1
                    oy2 = int(row[k]) + iy2
                break
        oranges.append(((ox1, ox2), (oy1, oy2)))
    return oranges

def input_output_abs(results):
    # 0: a, 1: 1, 2: 2
    oranges = []
    for row in results.itertuples():
        # for  output find appropriate range
        ox1 = int(row[5])
        ox2 = int(row[6])
        oy1 = int(row[7])
        oy2 = int(row[8])
        oranges.append(((ox1, ox2), (oy1, oy2)))
    return oranges

def query_comp(pranges, folder, tnames, absolute = False, merge = True, dtype = 'arrow'):
    con = duckdb.connect(database=':memory:')
    if dtype == 'arrow':
        tables = load_parquet(folder)
    else:
        raise ValueError('dtype not supported')
    

    for name in tnames:
        print(name)
        print(pranges)
        start = time.time()
        oranges = []
        if merge:
            pranges = merge_ranges(pranges)
        end = time.time()
        print(len(pranges))
        print('merge time: {}'.format(end - start))
        start = time.time()
        for prange in pranges:
            x1 = prange[0][0]
            x2 = prange[0][1]
            y1 = prange[1][0]
            y2 = prange[1][1]

            arrow_table = tables[name]
            print(arrow_table)
            df = con.execute("SELECT * FROM arrow_table WHERE LEAST(output_x2, {}) >= GREATEST(output_x1, {}) \
                AND LEAST(output_y2, {}) >= GREATEST(output_y1, {})".format(x2, x1, y2, y1)).fetchdf()
            print(df)
            if not absolute:
                oranges += input_output(prange, df)
            else:
                oranges += input_output_abs(df)
        if len(oranges) == 0:
            return oranges
        end = time.time()
        print('query time: {}'.format(end - start))
        pranges = oranges
    return pranges