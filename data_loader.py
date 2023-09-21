import os

import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_CIFAR_batch(filename):
    with open(filename,"rb") as f:
        datadict = unpickle(filename)
        data = datadict[b'data']
        lables = datadict[b"labels"]
        data = data.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
        lables = np.array(lables)
        return data,lables

def load_CIFAR10(ROOT):
    data_list = []
    lable_list = []
    for i in range(1,6):
        f = os.path.join(ROOT,"data_batch_%d" % (i))
        data,lables = load_CIFAR_batch(f)
        data_list.append(data)
        lable_list.append(lables)
    data_train = np.concatenate(data_list)
    lable_train = np.concatenate(lable_list)
    data_test,lable_test = load_CIFAR_batch(os.path.join(ROOT,"test_batch"))
    return data_train,lable_train,data_test,lable_test

