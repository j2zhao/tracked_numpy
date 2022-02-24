import numpy as np
import pandas as pd
import os
import openml


def preprocess():
    mnist_openml  = openml.datasets.get_dataset(554)
    Xy, _, _, _ = mnist_openml.get_data(dataset_format="array")
    return Xy

def split_train_test(data):
    test = data[1]
    return test


if __name__ == "__main__":
    #train = pd.read_csv("train.csv")
    #print(train.shape)
    train = preprocess()
    test = split_train_test(train)
    np.save("mnist.npy", test)