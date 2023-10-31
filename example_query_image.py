
import cv2
import pandas as pd
import pickle
import os
import numpy.core.tracked_float as tf
import numpy as np
import math

class DummyProv(object):
    def __init__(self, prov):
        self.provenance = prov

def array_upscale(arr, factor = 10):
    if len(arr.shape) == 2:
        x = arr.shape[0]*factor
        y = arr.shape[1]*factor
        arr_2 = np.zeros((x, y), dtype=arr.dtype)
        
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for a in range(factor):
                    for b in range(factor):
                        arr_2[(i*factor + a), (j*factor + b)] = arr[i, j]
        
        return arr_2
    else:
        x = arr.shape[0]*factor
        arr_2 = np.zeros((x), dtype=arr.dtype)
        
        for i in range(arr.shape[0]):
                for a in range(factor):
                    arr_2[(i*factor + a)] = arr[i]
        
        return arr_2

def convert(array):
    new_array = np.zeros(array.shape, dtype=object)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            new_array[i, j] = DummyProv(array[i, j].provenance)
    return new_array


def resize_aux(arr, w, h, prov = False):
    w_org = arr.shape[0]
    h_org = arr.shape[1]
    # if prov:
    #     new_arr = np.zeros((w, h), dtype=np.float64).astype(tf.tracked_float)
    # else:
    new_arr = np.zeros((w, h), dtype = object)
    
    for i in range(w):
        for j in range(h):
            x1 = math.floor(i*w_org/w)
            if i == w - 1:
                x2 = w_org
            else:
                x2 = math.floor((i + 1)*w_org/w)
            y1 = math.floor(j*h_org/h)
            if j == h - 1:
                y2 = h_org
            else:
                y2 = math.floor((j + 1)*h_org/h)
            total = (x2 - x1)*(y2 - y1)
            #new_arr[i, j] = np.sum(arr[x1:x2, y1:y2], initial= None)//total
            provenance = []
            for a in range(x1, x2):
                for b in range(y1, y2):
                    provenance.append((1, a, b))
            new_arr[i, j] = DummyProv(provenance)
    return new_arr


def resize_img(array, w = 416, h = 416):
    w_org = array.shape[0]
    h_org = array.shape[1]
    # prov_array = np.zeros((w_org, h_org)).astype(tf.tracked_float)
    # tf.initialize(prov_array, 1)
    prov_array = resize_aux(array, w, h, prov = True)
    new_array = np.zeros((w, h, 3))
    # for i in range(3):
    #     new_array[:, :, i] = resize_aux(array[:, :, i], w, h, prov = False)
    
    return new_array, prov_array

def lum_img(array, max = 255):
    w = array.shape[0]
    h = array.shape[1]
    prov_array = np.zeros((w, h)).astype(tf.tracked_float)
    tf.initialize(prov_array, 1)
    array = array + 1
    array = np.clip(array, a_min = 0, a_max = max)
    return array, prov_array

def rotate_img(array):
    w = array.shape[0]
    h = array.shape[1]
    prov_array = np.zeros((w, h)).astype(tf.tracked_float)
    tf.initialize(prov_array, 1)
    prov_array = np.rot90(prov_array, k=1, axes=(0, 1))
    array = np.rot90(array, k=1, axes=(0, 1))
    return array, prov_array

def flip_img(array):
    w = array.shape[0]
    h = array.shape[1]
    prov_array = np.zeros((w, h)).astype(tf.tracked_float)
    tf.initialize(prov_array, 1)
    prov_array = np.fliplr(prov_array)
    array = np.fliplr(array)
    return array, prov_array   

def lime_exp(afile, upscale = 10):
    arr = np.load(afile)
    if upscale != 0:
        arr = array_upscale(arr, factor = upscale)
    ar2 = np.zeros(arr.shape)
    ar2[arr > 0.5] = 1
    prov = np.empty(arr.shape,dtype = object)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if ar2[i, j] == 1:
                p = DummyProv([(1, i, j)])
                prov[i, j] = p
            else:
                p = DummyProv([])
                prov[i, j] = p
    return ar2, prov

if __name__ == '__main__': 
    folder = 'compression_tests_2/image_pipeline'
    image_dire = 'compression_tests_2/VIRAT_S_000101_10.jpeg'
    image = np.asarray(cv2.imread(image_dire))
    # Resize Image
    image, prov_arr = resize_img(image)
    #prov_arr = convert(prov_arr)
    #print(prov_arr[0,0].provenance)
    #raise ValueError()
    dire = os.path.join(folder, 'step1.npy')
    np.save(dire, prov_arr)
    # Change Luminosity
    image, prov_arr = lum_img(image)
    dire = os.path.join(folder, 'step2.npy')
    prov_arr = convert(prov_arr)
    np.save(dire, prov_arr)
    # Rotate Image
    image, prov_arr = rotate_img(image)
    dire = os.path.join(folder, 'step3.npy')
    prov_arr = convert(prov_arr)
    np.save(dire, prov_arr)
    # flip image
    image, prov_arr = flip_img(image)
    dire = os.path.join(folder, 'step4.npy')
    prov_arr = convert(prov_arr)
    np.save(dire, prov_arr)
    # Use Lime
    dire = os.path.join(folder, 'step5.npy')
    dire2 = os.path.join(folder, 'yolo_example.npy')
    image, prov_arr = lime_exp(dire2, upscale = 0)
    prov_arr = convert(prov_arr)
    np.save(dire, prov_arr)
    #dire = os.path.join(folder, 'lime_input.jpg')
    #cv2.imwrite(dire, image)
    #np.save(dire, image)