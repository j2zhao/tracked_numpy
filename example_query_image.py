
import cv2
import pandas as pd
import pickle
import os
import numpy.core.tracked_float as tf
import numpy as np

class DummyProv(object):
    def __init__(self, prov):
        self.provenance = prov

def convert(array, ids = [0]):
    new_array = np.zeros(array.shape, dtype=object)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            new_array[i, j] = DummyProv(array[i, j].provenance)
    return new_array


def resize_aux(arr, w, h, prov = False):
    w_org = arr.shape[0]
    h_org = arr.shape[1]
    x = w_org//w
    y = h_org//h
    if prov:
        new_arr = np.zeros((w, h)).astype(tf.tracked_float)
    else:
        new_arr = np.zeros((w, h))
    
    for i in range(w):
        for j in range(h):
            x1 = x*i
            if i == w -1:
                x2 = w_org
            else:
                x2 = x*(i + 1)
            y1 = y*j
            if j == h - 1:
                y2 = h_org
            else:
                y2 = y*(j + 1)
            new_arr[i, j] = np.sum(arr[x1:x2, y1:y2])
    
    return arr
def resize_img(array, w = 416, h = 416):
    w_org = array.shape[0]
    h_org = array.shape[1]
    prov_array = np.zeros((w_org, h_org)).astype(tf.tracked_float)
    tf.initialize(prov_array, 1)
    prov_array = resize_aux(prov_array, w, h, prov = True)

    new_array = np.zeros((w, h, 3))
    for i in range(3):
        new_array[:, :, i] = resize_aux(array[:, :, i], w, h, prov = False)
    
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

if __name__ == '__main__': 
    folder = ''
    image_dire = ''
    image = np.asarray(cv2.imread(image_dire))
    
    # Resize Image - TODO
    image, prov_arr = resize_img(image)
    dire = os.path.join(folder, 'step1.npy')
    np.save(dire, prov_arr)
    # Change Luminosity - TODO
    image, prov_arr = lum_img(image)
    dire = os.path.join(folder, 'step2.npy')
    np.save(dire, prov_arr)
    # Rotate Image - TODO
    image, prov_arr = rotate_img(image)
    dire = os.path.join(folder, 'step3.npy')
    np.save(dire, prov_arr)
    # flip image - TODO
    image, prov_arr = rotate_img(image)
    dire = os.path.join(folder, 'step4.npy')
    np.save(dire, prov_arr)
    # Use Lime
    dire = os.path.join(folder, 'lime_input.npy')
    np.save(dire, image)