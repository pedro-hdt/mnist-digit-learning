import numpy as np
import scipy.io as sio
from montage import *


def task1_2(X, Y):
    # Input:
    # X : M-by-D data matrix (double)
    # Y : M-by-1 label vector (uint8)
    # Output:
    # M : (K+1)-by-D mean vector matrix (double)
    # Note that M[K+1, :] is the mean vector of X
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

    M[10] = np.sum(M[:9,:], axis=0) / 10

    montage(M)
    # TODO: Before submitting uncomment the following line
    # plt.show()

    return M



