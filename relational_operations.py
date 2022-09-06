import pandas as pd
import numpy as np
from tracked_object import TrackedObj
import pickle

def groupby_prov(df, col_name, agg_name, id1 = 1, id2 = 2, limit = 1000):
    """
    Sum group by with col_name, on aggregation_name
    """
    col1 = []
    col1_dict = {}
    col2 = []
    columns = list(df.columns)
    columns_dict = {}
    for i, c in enumerate(columns):
        columns_dict[c] = i

    for index, row in df.iterrows():
        if index >= limit:
            break
        if row[col_name] in col1_dict:
            i = col1_dict[row[col_name]]
            col2[i] += TrackedObj(row[agg_name], (id2, columns_dict[agg_name], index))
        else:
            i = len(col1)
            col1_dict[row[col_name]] = i
            obj1 = TrackedObj(row[col_name], (id1, columns_dict[col_name], index))
            obj2 = TrackedObj(row[agg_name], (id2, columns_dict[agg_name], index))
            col1.append(obj1)
            col2.append(obj2)
    return np.asarray([col1, col2], dtype=object)



def join_prov(df1, df2, column_name1, column_name2,  id1 = 1, id2 = 2, limit = 100000):
    """
    Inner join on column_name over dataset1 and dataset 2
    """
    df1 = df1.head(limit)
    df2 = df2.head(limit)
    array = []
    col1 = list(df1.columns)
    print(col1)
    col1_dict = {}
    for i, c in enumerate(col1):
        col1_dict[c] = i
    col2 = list(df2.columns)
    print(col2)
    col2_dict = {}
    for i, c in enumerate(col2):
        col2_dict[c] = i

    u1 = set(df1[column_name1].unique())
    u2 = set(df2[column_name2].unique())
    u = u1.intersection(u2)
    print(len(u))
    a = 0
    for val in u:
        if a%1000 == 0:
            print(val)
            print(a)
        df1_sub = df1.loc[df1[column_name1] == val]
        df2_sub = df2.loc[df2[column_name2] == val]
        for i, row1 in df1_sub.iterrows():
            for j, row2 in df2_sub.iterrows():
                row = []
                for name, value in row1.items():
                    obj = TrackedObj(value, (id1, col1_dict[name], i))
                    row.append(obj)

                for name, value in row2.items():
                    if name == column_name2:
                        continue
                    obj = TrackedObj(value, (id2, col2_dict[name], j))
                    row.append(obj)
                
                array.append(row)
        a += 1
    return np.asarray(array, dtype=object)
                

if __name__ == "__main__":
    data = 'group_by_pandas_sorted.pickle'
    col_name = 'startYear'
    agg_name = 'isAdult'
    data = open(data, 'rb')
    data = pickle.load(data, encoding='latin1')
    data = groupby_prov(data, col_name, agg_name)

