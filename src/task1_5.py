from my_kMeansClustering import *
import scipy.io as sio
from time import time
from montage import *
from scipy.cluster.vq import kmeans # TODO: remove this import after testing
import hashlib


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
    libCs = []

    # Do all computation because saving is asynchronous which makes it dangerous
    for k in Ks:

        C, idx, SSE = my_kMeansClustering(X, k, X[:k])
        libC, _ = kmeans(X, k_or_guess=X[:k], iter=1)
        libCs.append(np.copy(libC))
        Cs.append(np.copy(C))
        idxs.append(np.copy(idx))
        SSEs.append(np.copy(SSE))

    # Save all files
    for i in range(len(Ks)):

        k = Ks[i]
        sio.savemat(file_name='task1_5_c_{}.mat'.format(k), mdict={'C': Cs[i]})
        sio.savemat(file_name='task1_5_idx_{}.mat'.format(k), mdict={'idx': idxs[i]})
        sio.savemat(file_name='task1_5_sse_{}.mat'.format(k), mdict={'SSE': SSEs[i]})
        sio.savemat(file_name='lib{}.mat'.format(k), mdict={'C': libCs[i]})

    for k in Ks:
        hasher = hashlib.md5()
        with open('task1_5_c_{}.mat'.format(k)) as f:
            hasher.update(f.read())
            result = hasher.hexdigest()

        with open('inside{}.txt'.format(k), 'w+') as f:
            f.write(result)

    for i in range(len(Ks)):

        libC = sio.loadmat(file_name='lib{}.mat'.format(Ks[i]))['C']
        C = sio.loadmat(file_name='task1_5_c_{}.mat'.format(Ks[i]))['C']

        print np.allclose(libC, C)

        # visualise
        montage(C)
        plt.suptitle('Your result')
        plt.figure()
        montage(libC)
        plt.suptitle('Library function')
        plt.show()
