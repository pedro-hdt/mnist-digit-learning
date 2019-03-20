import numpy as np
import scipy.io as sio
from montage import *


def task1_2(X, Y):

    """ Write a Python function that calculates a mean vector of data for each class (k = 1, . . . , K,
    where K = 10) and for all, and displays the images of K + 1 mean vectors in a single graph
    using montage() function.
    Inputs:
    X and Y: the same formats as in Task 1.1.
    M: (K+1)-by-D mean vector matrix (float64),
    where K (not an input) is the number of classes, and D is the same as in Task 1.1.
    M(K+1,:) is the mean vector of the whole data."""


    D = len(X[0])
    M = np.zeros((11, D))


    # for each of the 10 classes
    for i in range(10):
        samples = 0
        # for each data'point'
        for j in range(len(X)):
            if Y[j] == i:
                M[Y[j]] += X[j]
                samples += 1
        M[i] /= samples

    # TODO: check the total mean is mean of means
    M[10] = np.sum(M[:9, :], axis=0) / 10

    montage(M)
    # TODO: Before submitting uncomment the following line
    # plt.show()

    return M



