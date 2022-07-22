import numpy as np
import pandas as pd
import os
import openml
import cv2



def preprocess():
    mnist_openml  = openml.datasets.get_dataset(554)
    Xy, _, _, _ = mnist_openml.get_data(dataset_format="array")
    return Xy

def split_train_test(data):
    test = data[1,:-1]
    return test
def upsize(data, size = (1000,1000)):
    data = np.reshape(data, (28,28))
    cv2.imshow("Input", data)
    resized = cv2.resize(data, (1000,1000))
    img = np.reshape(resized, (-1, ))
    return img

if __name__ == "__main__":
    #train = pd.read_csv("train.csv")
    #print(train.shape)
    img = preprocess()
    img = split_train_test(img)
    img = upsize(img)
    np.save("mnist_2.npy", img)