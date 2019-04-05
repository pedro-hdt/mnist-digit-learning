from my_dist import *
from scipy.stats import mode
import sys
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

    # redirect output from stdout to text file so we can use the results in report
    # sys.stdout = open('../results/task2_1_log.txt', 'w+')

    N = len(Xtst)
    L = len(Ks)
    Ypreds = np.zeros((N, L), dtype='uint8')
    Ypreds2 = np.zeros((N, L), dtype='uint8')


    # Compute distance from every point in test set to every point in dataset
    dist_mat = vec_sq_dist(Xtrn, Xtst)

    start_time = time()
    for l in range(L):
        k = Ks[l] # k in k-NN

        # # get sorting index for distances to first k-nn only
        # idx = dist_mat.argsort(axis=0)[:k]
        #
        # # extract list of closest training samples
        # k_nn_labels = Ytrn[idx]
        #
        # # we predict it is the most common value (ie the mode) of the k-nn
        # Ypreds2[:, l], _ = mode(k_nn_labels)


        for n in range(N):

            # distance vector between test sample and each of training samples
            dist = dist_mat[:, n]

            # get sorting index for distances to first k-nn only
            idx = dist.argsort()[:k]

            # extract list of closest training samples
            k_nn_labels = Ytrn[idx]

            # we predict it is the most common value (ie the mode) of the k-nn
            Ypreds[n, l], _ = mode(k_nn_labels)

            #print np.array_equal(Ypreds[n, l], Ypreds2[n, l].T)

    runtime = time()- start_time
    print runtime

    return Ypreds