import torch
import numpy as np
import torch.nn.functional as F
import torch.nn
from torch.autograd import Variable
import os.path
import scipy
from scipy import io
import sys

class data_test(torch.utils.data.Dataset):
    def __init__(self, transform=False, class_balancing=False, index=0):
        self.transform = transform

        stringX = 'x_d' + str(index) + '.mat'
        X_str = 'x_d' + str(index)

        stringY_L = 'y_dL' + str(index) + '.mat'
        Y_str_L = 'y_L' + str(index)

        stringY_Fi = 'y_dFi' + str(index) + '.mat'
        Y_str_Fi = 'y_Fi' + str(index)

        stringY_Fo = 'y_dFo' + str(index) + '.mat'
        Y_str_Fo = 'y_Fo' + str(index)

        stringY_T = 'y_dT' + str(index) + '.mat'
        Y_str_T = 'y_T' + str(index)

        x = scipy.io.loadmat(stringX, mdict=None)
        x = x[X_str]

        y_L = scipy.io.loadmat(stringY_L, mdict=None)
        y_L = y_L[Y_str_L]

        y_Fi = scipy.io.loadmat(stringY_Fi, mdict=None)
        y_Fi = y_Fi[Y_str_Fi]

        y_Fo = scipy.io.loadmat(stringY_Fo, mdict=None)
        y_Fo = y_Fo[Y_str_Fo]

        y_T = scipy.io.loadmat(stringY_T, mdict=None)
        y_T = y_T[Y_str_T]

        self.X = torch.FloatTensor(np.expand_dims(x, 1).astype(np.float32))
        self.Y_L = torch.tensor(y_L, dtype=torch.long)
        self.Y_Fi = torch.tensor(y_Fi, dtype=torch.long)
        self.Y_Fo = torch.tensor(y_Fo, dtype=torch.long)
        self.Y_T = torch.tensor(y_T, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = [self.X[idx], self.Y_L[idx], self.Y_Fi[idx], self.Y_Fo[idx], self.Y_T[idx]]
        if self.transform:
            sample[0] = self.transform(sample[0])
        return sample


class data_train(torch.utils.data.Dataset):
    def __init__(self, transform=False, class_balancing=False, index=0, fold=1):
        self.transform = transform

        folds_index = range(fold)
        folds_index = [x + 1 for x in folds_index]
        del [folds_index[index - 1]]
        count = 0
        for i in folds_index:
            stringX = 'x_d' + str(i) + '.mat'
            X_str = 'x_d' + str(i)

            stringY_L = 'y_dL' + str(i) + '.mat'
            Y_str_L = 'y_L' + str(i)

            stringY_Fi = 'y_dFi' + str(i) + '.mat'
            Y_str_Fi = 'y_Fi' + str(i)

            stringY_Fo = 'y_dFo' + str(i) + '.mat'
            Y_str_Fo = 'y_Fo' + str(i)

            stringY_T = 'y_dT' + str(i) + '.mat'
            Y_str_T = 'y_T' + str(i)

            x_partial = scipy.io.loadmat(stringX, mdict=None)
            x_partial = x_partial[X_str]

            y_partial_L = scipy.io.loadmat(stringY_L, mdict=None)
            y_partial_L = y_partial_L[Y_str_L]

            y_partial_Fi = scipy.io.loadmat(stringY_Fi, mdict=None)
            y_partial_Fi = y_partial_Fi[Y_str_Fi]

            y_partial_Fo = scipy.io.loadmat(stringY_Fo, mdict=None)
            y_partial_Fo = y_partial_Fo[Y_str_Fo]

            y_partial_T = scipy.io.loadmat(stringY_T, mdict=None)
            y_partial_T = y_partial_T[Y_str_T]

            if count == 0:
                x = x_partial
                y_L = y_partial_L
                y_Fi = y_partial_Fi
                y_Fo = y_partial_Fo
                y_T = y_partial_T

            else:
                x = np.concatenate((x, x_partial), axis=0)
                y_L = np.concatenate((y_L, y_partial_L), axis=0)
                y_Fi = np.concatenate((y_Fi, y_partial_Fi), axis=0)
                y_Fo = np.concatenate((y_Fo, y_partial_Fo), axis=0)
                y_T = np.concatenate((y_T, y_partial_T), axis=0)

            count = count + 1

        self.X = torch.FloatTensor(np.expand_dims(x, 1).astype(np.float32))
        self.Y_L = torch.tensor(y_L, dtype=torch.long)
        self.Y_Fi = torch.tensor(y_Fi, dtype=torch.long)
        self.Y_Fo = torch.tensor(y_Fo, dtype=torch.long)
        self.Y_T = torch.tensor(y_T, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = [self.X[idx], self.Y_L[idx], self.Y_Fi[idx], self.Y_Fo[idx], self.Y_T[idx]]
        if self.transform:
            sample[0] = self.transform(sample[0])
        return sample