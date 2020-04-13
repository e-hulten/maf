import torch
import numpy as np
import matplotlib.pyplot as plt


class MINIBOONE:
    class Data:
        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):

        file = "data/maf_data/miniboone/data.npy"
        trn, val, tst = load_data_normalised(file)

        self.train = self.Data(trn)
        self.val = self.Data(val)
        self.test = self.Data(tst)

        self.n_dims = self.train.x.shape[1]


def load_data(root_path):

    data = np.load(root_path)
    N_test = int(0.1 * data.shape[0])
    data_test = data[-N_test:]
    data = data[0:-N_test]
    N_validate = int(0.1 * data.shape[0])
    data_validate = data[-N_validate:]
    data_train = data[0:-N_validate]

    return data_train, data_validate, data_test


def load_data_normalised(root_path):

    data_train, data_validate, data_test = load_data(root_path)
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu) / s
    data_validate = (data_validate - mu) / s
    data_test = (data_test - mu) / s

    return data_train, data_validate, data_test

