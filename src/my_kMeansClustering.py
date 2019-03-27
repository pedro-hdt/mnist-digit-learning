import numpy as np
from my_mean import *
from my_dist import *


def my_kMeansClustering(X, k, initialCentres, maxIter=500):
    """ Write a Python function that carries out the k-means clustering

    Input
    X : N-by-D matrix (double) of input sample data
    k : scalar (integer) - the number of clusters
    initialCentres : k-by-D matrix (double) of initial cluster centres
    maxIter  : scalar (integer) - the maximum number of iterations

    Output
    C   : k-by-D matrix (double) of cluster centres
    idx : N-by-1 vector (integer) of cluster index table
    SSE : (L+1)-by-1 vector (double) of sum-squared-errors """

    # TODO: remove printing

    N = len(X)

    idx = np.zeros((N, ))
    idx_prev = np.zeros((N, ))
    C = initialCentres

    dist = np.zeros((k, N))

    # initialise error list
    SSE = []

    # show cluster centres at iteration 0
    # print "[0] Iteration: ", C

    # Compute Squared Euclidean distance (i.e. the squared distance)
    # between each cluster centre and each observation
    for i in range(maxIter):

        for cluster in range(k):
            dist[cluster] = sq_dist(X, C[cluster]) # TODO this dist can be used for error. Why are we recalculating?

        # Assign data to clusters
        # Ds are the actual distances and idx are the cluster assignments
        idx = dist.argmin(axis=0).T  # find min dist. for each observation

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
        # SSE.append(sum_sq_error(X, cluster, C, idx, N))
        SSE.append(sum_sq_error2(dist, k, idx, N))

        # If assignments were maintained, terminate
        if np.array_equal(idx, idx_prev):
            return C, idx, np.array(SSE)

        # Store current assignment to check if it changes in next iteration
        idx_prev = idx

    return C, idx, np.array(SSE)


def sum_sq_error(X, k, centres, idx, N):

    error = 0

    for cluster in range(k):
        class_k = X[idx[:] == cluster]
        error += np.sum(sq_dist(class_k, centres[cluster]))

    error *= (1. / N)

    return error
