from my_kMeansClustering import *
import scipy.io as sio
from time import time


def task1_5(X, Ks):
    """ Write a Python function that calls the k-means clustering function for each k in a vector Ks,
    using the first k samples in the data set X as the initial cluster centres, and saves the returned
    C, idx, and SSE as 'task1_5_c_{k}.mat', 'task1_5_idx_{k}.mat', and 'task1_5_sse_{k}.mat', respectively.

    Input:
    X : M-by-D data matrix (double)
    Ks : 1-by-L vector (integer) of the numbers of nearest neighbours """

    n = len(Ks)
    C = []
    idx = []
    SSE = []

    for k in range(n):
        C_k, idx_k, SSE_k = my_kMeansClustering(X, Ks[k], X[:Ks[k]])
        C.append(C_k)
        idx.append(idx_k)
        SSE.append(SSE_k)

    # TODO: remove timing code
    start_time = time()
    # Saving the data
    for k in range(n):
        sio.savemat(file_name='../results/task1_5_c_{}.mat'.format(Ks[k]), mdict={'C': C[k]})
        sio.savemat(file_name='../results/task1_5_idx_{}.mat'.format(Ks[k]), mdict={'idx': idx[k]})
        sio.savemat(file_name='../results/task1_5_sse_{}.mat'.format(Ks[k]), mdict={'SSE': SSE[k]})

    print 'Saving the data took: {} secs'.format(time()-start_time)

