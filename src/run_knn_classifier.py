from my_dist import *
from scipy.stats import mode
from time import time

def run_knn_classifier(Xtrn, Ytrn, Xtst, Ks):
    """
    Write a Python function for classification with k-NN

    :param Xtrn: M-by-D training data matrix
    :param Ytrn: M-by-1 label vector for Xtrn
    :param Xtst: N-by-D test data matrix
    :param Ks: L-by-1 vector of the numbers of nearest neighbours in Xtrn
    :return: Ypreds : N-by-L matrix of predicted labels for Xtst
    """

    N = len(Xtst)
    L = len(Ks)
    Ypreds = np.zeros((N, L), dtype='uint8')

    start_time = time()

    # Compute distance from every point in test set to every point in dataset
    dist_mat = vec_sq_dist(Xtrn, Xtst)

    # get sorting index for all distances
    idx = dist_mat.argsort(axis=0)

    overhead_runtime = time() - start_time
    print 'Overhead runtime: ', overhead_runtime

    for l in range(L):

        start_time = time()

        k = Ks[l] # k in k-NN

        for n in range(N):

            # extract list of closest training samples using the first k entries of the index we calculated
            k_nn_labels = Ytrn[idx[:k, n]]

            # we predict it is the most common value (ie the mode) of the k-nn
            Ypreds[n, l], _ = mode(k_nn_labels)

        runtime = time() - start_time + overhead_runtime
        print '\nRuntime of k-NN for k = {}'.format(Ks[l])
        print runtime

    return Ypreds
