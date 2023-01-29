import numpy as np
import os
def to_column_1(array, temp_path, zeros = True):
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    a = 0
    b = 0
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            provenance = array[i, j].provenance
            for id, x, y in provenance:
                x1.append(i)
                x2.append(j)
                y1.append(x)
                y2.append(y)
    
    x1 = np.array(x1).astype(np.intc).flatten(order='F')
    #print(x1.shape)
    path = os.path.join(temp_path, 'x1.csv')
    #np.save(path, x1)
    x1.tofile(path, sep = ',')

    x2 = np.array(x2).astype(np.intc).flatten(order='F') 
    #print(x2.shape)
    path = os.path.join(temp_path, 'x2.csv')
    #np.save(path, x2)
    x2.tofile(path, sep = ',')

    y1 = np.array(y1).astype(np.intc).flatten(order='F') 
    #print(y1.shape)
    path = os.path.join(temp_path, 'y1.csv')
    np.save(path, y1)
    y1.tofile(path, sep = ',')

    y2 = np.array(y2).astype(np.intc).flatten(order='F')
    #print(y2.shape)
    path = os.path.join(temp_path, 'y2.csv')
    np.save(path, y2)
    y2.tofile(path, sep = ',')

def to_column_2(array, temp_path, input_id = (1, 2), zeros= True):
    x1 = []
    x2 = []
    y1 = []
    y2 = []

    w1 = []
    w2 = []
    z1 = []
    z2 = []
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if zeros:
                if len(array[i, j].provenance) == 0:
                    continue
            provenance = array[i, j].provenance
            for id, x, y in provenance:
                if id == input_id[0]:
                    x1.append(i)
                    x2.append(j)
                    y1.append(x)
                    y2.append(y)
                else:
                    w1.append(i)
                    w2.append(j)
                    z1.append(x)
                    z2.append(y)
    x1 = np.array(x1).astype(np.intc).flatten(order='F') 
    path = os.path.join(temp_path, 'x1.npy')
    #np.save(path, x1)
    x1.tofile(path, sep = ',')

    x2 = np.array(x2).astype(np.intc).flatten(order='F') 
    path = os.path.join(temp_path, 'x2.csv')
    #np.save(path, x2)
    x2.tofile(path, sep = ',')

    y1 = np.array(y1).astype(np.intc).flatten(order='F') 
    path = os.path.join(temp_path, 'y1.csv')
    #np.save(path, y1)
    y1.tofile(path, sep = ',')

    y2 = np.array(y2).astype(np.intc).flatten(order='F') 
    path = os.path.join(temp_path, 'y2.csv')
    #np.save(path, y2)
    y2.tofile(path, sep = ',')

    w1 = np.array(w1).astype(np.intc).flatten(order='F') 
    path = os.path.join(temp_path, 'w1.csv')
    #np.save(path, w1)
    w1.tofile(path, sep = ',')

    w2 = np.array(w2).astype(np.intc).flatten(order='F') 
    path = os.path.join(temp_path, 'w2.csv')
   #np.save(path, w2)
    w2.tofile(path, sep = ',')

    z1 = np.array(z1).astype(np.intc).flatten(order='F') 
    path = os.path.join(temp_path, 'z1.csv')
    #np.save(path, z1)
    z1.tofile(path, sep = ',')

    z2 = np.array(z2).astype(np.intc).flatten(order='F') 
    path = os.path.join(temp_path, 'z2.csv')
    #np.save(path, z2)
    z2.tofile(path, sep = ',')