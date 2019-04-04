from my_kMeansClustering import *
import scipy.io as sio
from montage import *


def task1_5(X, Ks):
    """
    Write a Python function that calls the k-means clustering function for each k in a vector Ks,
    using the first k samples in the data set X as the initial cluster centres, and saves the returned
    C, idx, and SSE as 'task1_5_c_{k}.mat', 'task1_5_idx_{k}.mat', and 'task1_5_sse_{k}.mat', respectively.

    :param X: M-by-D data matrix (double)
    :param Ks: 1-by-L vector (integer) of the numbers of nearest neighbours
    """

    Cs = []
    idxs = []
    SSEs = []

    # Do all computation because saving is asynchronous which makes it dangerous
    for k in Ks:
        C, idx, SSE = my_kMeansClustering(X, k, X[:k])
        Cs.append(np.copy(C))
        idxs.append(np.copy(idx))
        SSEs.append(np.copy(SSE))

    # Save all files
    for i in range(len(Ks)):
        k = Ks[i]
        sio.savemat(file_name='task1_5_c_{}.mat'.format(k), mdict={'C': Cs[i]})
        sio.savemat(file_name='task1_5_idx_{}.mat'.format(k), mdict={'idx': idxs[i]})
        sio.savemat(file_name='task1_5_sse_{}.mat'.format(k), mdict={'SSE': SSEs[i]})
