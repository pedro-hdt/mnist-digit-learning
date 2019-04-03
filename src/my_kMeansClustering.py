import numpy as np
from my_mean import *
from my_dist import *


def my_kMeansClustering(X, k, initialCentres, maxIter=500):
    """
    Write a Python function that carries out the k-means clustering and returns

    :param X: N-by-D matrix (double) of input sample data
    :param k: scalar (integer) - the number of clusters
    :param initialCentres: k-by-D matrix (double) of initial cluster centres
    :param maxIter: scalar (integer) - the maximum number of iterations
    :returns: tuple (C, idx, SSE)

    - C - k-by-D matrix (double) of cluster centres
    - idx - N-by-1 vector (integer) of cluster index table
    - SSE - (L+1)-by-1 vector (double) of sum-squared-errors
    """


    # TODO: remove printing?

    N = len(X)

    idx = np.zeros((N, ))
    idx_prev = np.zeros((N, ))
    C = initialCentres

    # initialise error list
    SSE = np.zeros(maxIter)

    # show cluster centres at iteration 0
    # print "[0] Iteration: ", C

    # Compute Squared Euclidean distance (i.e. the squared distance)
    # between each cluster centre and each observation
    for i in range(maxIter):

        dist = vec_sq_dist(X, C)

        # Assign data to clusters
        # Ds are the actual distances and idx are the cluster assignments
        idx = dist.argmin(axis=1).T  # find min dist. for each observation

        # Update cluster centres
        for cluster in range(k):
            # check the number of samples assigned to this cluster
            if np.sum(idx == cluster) == 0:
                print('k-means: cluster {} is empty'.format(cluster))
            else:
                class_k = X[idx[:] == cluster]
                C[cluster] = my_mean(class_k)

        # show cluster centres at iteration i
        # print '[{}] Iteration: '.format(i+1), C

        # add error to the list
        SSE[i] = sum_sq_error(dist, k, idx, N)

        # If assignments were maintained, terminate
        if np.array_equal(idx, idx_prev):
            return C, idx, SSE[:i+1]

        # Store current assignment to check if it changes in next iteration
        idx_prev = idx

    return C, idx, SSE


def sum_sq_error(dist, k, idx, N):

    error = 0

    for cluster in range(k):
        error += np.sum(dist[idx[:] == cluster, cluster])

    error *= (1. / N)

    return error
