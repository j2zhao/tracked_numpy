
#import cv2
import pandas as pd
import pickle
import os
import numpy.core.tracked_float as tf
import numpy as np
import math
import copy
from compression import compression
from compression_algs import convert_rel, convert_inverse_rel

class DummyProv(object):
    def __init__(self, prov):
        self.provenance = prov

def convolution(image):
    # Dimensions of the input image
    image_rows, image_cols = image.shape
    # Padding the image
    arr = image.astype(tf.tracked_float)
    tf.initialize(arr, 1)
    padded_image = np.pad(arr, ((1,1),(1,1)), 'constant', constant_values=0)
    # Dimensions of the output image
    output_rows = image_rows
    output_cols = image_cols
    # Initializing the output image
    convoluted_image = np.zeros((output_rows, output_cols)).astype(tf.tracked_float)

    # Apply convolution
    for i in range(output_rows):
        for j in range(output_cols):
            #total_sum = 
            for a in range(3):
                for b in range(3):
                    convoluted_image[i,j] += padded_image[i + a, j + b]
    return convoluted_image

def convert(array):
    new_array = np.zeros(array.shape, dtype=object)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            new_array[i, j] = DummyProv(array[i, j].provenance)
    return new_array

def batchnorm(array):
    '''
    Return identity provenance to simulate batch size of 1
    '''
    arr = array.astype(tf.tracked_float)
    tf.initialize(arr, 1)
    new_array = np.zeros(array.shape, dtype=object)
    
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            # provenance = []
            # for a in range(array.shape[0]):
            #     for b in range(array.shape[1]):
            #         provenance.append((1, a, b))
            #print(j)
            #prov2 = copy.deepcopy(provenance)
            new_array[i, j] = DummyProv(arr[i, j].provenance)
    return new_array

def relu(array):
    '''
    Return identity provenance to simulate relu provenance
    '''
    arr = array.astype(tf.tracked_float)
    tf.initialize(arr, 1)
    new_array = np.zeros(array.shape, dtype=object)
    
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            new_array[i, j] = DummyProv(arr[i, j].provenance)
    return new_array

def array_sum(array1, array2):
    '''
    Return  provenance to simulate addition
    '''
    arr1 = array1.astype(tf.tracked_float)
    tf.initialize(arr1, 1)
    arr2 = array2.astype(tf.tracked_float)
    tf.initialize(arr2, 2)
    new_array = np.zeros(array1.shape, dtype=object)
    
    for i in range(array1.shape[0]):
        for j in range(array2.shape[1]):
            new_array[i, j] = DummyProv(arr1[i, j].provenance + arr2[i, j].provenance)
    return new_array

def conv11(array):
    '''
    Return identity provenance to simulate 1x1 convolution
    '''
    arr = array.astype(tf.tracked_float)
    tf.initialize(arr, 1)
    new_array = np.zeros(array.shape, dtype=object)
    
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            new_array[i, j] = DummyProv(arr[i, j].provenance)
    return new_array

if __name__ == '__main__': 
    folder = 'compression_tests_2/resnet_pipeline'
    image_dire = 'compression_tests_2/VIRAT_S_000101_10.jpeg'
    #image = np.asarray(cv2.imread(image_dire))
    #image = image[:, :, 0]
    #print(image.shape)
    image = np.zeros((1080,1920)) #(1080, 1920)
    # apply 3x3 convolution
    image = convolution(image)
    prov_arr = convert(image)
    dire = os.path.join(folder, 'step1.npy')
    np.save(dire, prov_arr)

    # apply batch norm
    image = np.zeros(image.shape)
    image = batchnorm(image)
    prov_arr = convert(image) 
    dire = os.path.join(folder, 'step2.npy')
    np.save(dire, prov_arr)

    # Apply ReLu
    image = np.zeros(image.shape)
    image = relu(image)
    prov_arr = convert(image) 
    dire = os.path.join(folder, 'step3.npy')
    np.save(dire, prov_arr)

    # Apply 3x3 convolution
    image = np.zeros(image.shape)
    image = convolution(image)
    prov_arr = convert(image) 
    dire = os.path.join(folder, 'step4.npy')
    np.save(dire, prov_arr)

    # Apply batch norm
    image = np.zeros(image.shape)
    image = batchnorm(image)
    prov_arr = convert(image) 
    dire = os.path.join(folder, 'step5.npy')
    np.save(dire, prov_arr)

    # apply sum 
    image1 = np.zeros(image.shape)
    image2 = np.zeros(image.shape)
    image = array_sum(image1, image2)
    prov_arr = convert(image) 
    dire = os.path.join(folder, 'step6.npy')
    np.save(dire, prov_arr)

    # apply relu (1)
    image = np.zeros(image.shape)
    image = relu(image)
    prov_arr = convert(image) 
    dire = os.path.join(folder, 'step7.npy')
    np.save(dire, prov_arr)


