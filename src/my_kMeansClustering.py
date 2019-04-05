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
    - SSE - (L+1)-by-1 vector (double) of sum-squared-errors where L is the number of iterations done
    """

    N = len(X)

    idx = np.zeros((N, 1))
    idx_prev = np.zeros((N, 1))
    C = initialCentres

    # initialise error list
    SSE = np.zeros((maxIter+1, 1))

    # Compute Squared Euclidean distance (i.e. the squared distance)
    # between each cluster centre and each observation
    for i in range(maxIter):

        dist = vec_sq_dist(X, C)

        # Assign data to clusters
        # Ds are the actual distances and idx are the cluster assignments
        idx = dist.argmin(axis=1).T  # find min dist. for each observation

        # add error to the list
        SSE[i] = sum_sq_error(dist, k, idx, N)

        # Update cluster centres
        for cluster in range(k):
            # check the number of samples assigned to this cluster
            if np.sum(idx == cluster) == 0:
                print('k-means: cluster {} is empty'.format(cluster))
            else:
                class_k = X[idx[:] == cluster]
                C[cluster] = my_mean(class_k)

        # If assignments were maintained, terminate
        if np.array_equal(idx, idx_prev):
            # but first add final error to the list
            SSE[i+1] = sum_sq_error(dist, k, idx, N)
            return C, idx, SSE[:i+2]

        # Store current assignment to check if it changes in next iteration
        idx_prev = idx

    # add final error to the list
    SSE[-1] = sum_sq_error(dist, k, idx, N)

    return C, idx, SSE


def sum_sq_error(dist, k, idx, N):
    # TODO make this one more efficient
    error = 0

    for cluster in range(k):
        error += np.sum(dist[idx[:] == cluster, cluster])

    error *= (1. / N)

    return error
