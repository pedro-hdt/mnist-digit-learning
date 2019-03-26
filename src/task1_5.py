from my_kMeansClustering import *
import scipy.io as sio


def task1_5(X, Ks):
    # Input:
    # X : M-by-D data matrix (double)
    # Ks : 1-by-L vector (integer) of the numbers of nearest neighbours

    for k in Ks:
        C, idx, SSE = my_kMeansClustering(X, k, X[:k])
        sio.savemat(file_name='../results/task1_5_c_{}.mat'.format(k), mdict={'C': C})
        sio.savemat(file_name='../results/task1_5_idx_{}.mat'.format(k), mdict={'idx': idx})
        sio.savemat(file_name='../results/task1_5_sse_{}.mat'.format(k), mdict={'SSE': SSE})

