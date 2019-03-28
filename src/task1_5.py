from my_kMeansClustering import *
import scipy.io as sio
from time import time
from montage import *
from scipy.cluster.vq import kmeans # TODO: remove this import after testing


def task1_5(X, Ks):
    """ Write a Python function that calls the k-means clustering function for each k in a vector Ks,
    using the first k samples in the data set X as the initial cluster centres, and saves the returned
    C, idx, and SSE as 'task1_5_c_{k}.mat', 'task1_5_idx_{k}.mat', and 'task1_5_sse_{k}.mat', respectively.
    Input:
    :param X: M-by-D data matrix (double)
    :param Ks: 1-by-L vector (integer) of the numbers of nearest neighbours """

    for k in Ks:
        C_k, idx_k, SSE_k = my_kMeansClustering(X, k, X[:k])
        sio.savemat(file_name='task1_5_c_{}.mat'.format(k), mdict={'C': C_k})

        # Testing
        # loaded = sio.loadmat(file_name='task1_5_c_{}.mat'.format(k))['C']
        # assert np.array_equal(C_k, loaded)
        # C, distortion = kmeans(X, k_or_guess=X[:k], iter=1)
        # plt.clf()
        # montage(np.concatenate((C, np.zeros_like(C), loaded)))
        # plt.suptitle('Lib (first), You (second)')
        # plt.show()

        sio.savemat(file_name='task1_5_idx_{}.mat'.format(k), mdict={'idx': idx_k})
        sio.savemat(file_name='task1_5_sse_{}.mat'.format(k), mdict={'SSE': SSE_k})

