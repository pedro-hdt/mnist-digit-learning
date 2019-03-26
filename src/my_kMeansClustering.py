import numpy as np
from my_mean import *


def my_kMeansClustering(X, k, initialCentres, maxIter=500):
    # Input
    # X : N-by-D matrix (double) of input sample data
    # k : scalar (integer) - the number of clusters
    # initialCentres : k-by-D matrix (double) of initial cluster centres
    # maxIter  : scalar (integer) - the maximum number of iterations
    # Output
    # C   : k-by-D matrix (double) of cluster centres
    # idx : N-by-1 vector (integer) of cluster index table
    # SSE : (L+1)-by-1 vector (double) of sum-squared-errors

    N = len(X)
    D = X[0].size

    idx = np.empty(N, 1)
    C = np.empty(k, D)

    dist = np.zeros((k, N))

    idx_prev = np.empty(N, 1)

    # initialise error list
    SSE = np.empty()

    # show cluster centres at iteration 0
    print("[0] Iteration: ", C.tolist())

    # Compute Squared Euclidean distance (i.e. the squared distance)
    # between each cluster centre and each observation
    for i in range(maxIter):
        for c in range(k):
            dist[c] = sq_dist(X, C[:, c])

        # Assign data to clusters
        # Ds are the actual distances and idx are the cluster assignments
        idx = dist.argmin(axis=0)  # find min dist. for each observation

        # Update cluster centres
        for c in range(k):
            # check the number of samples assigned to this cluster
            if (np.sum(idx == c) == 0):
                print('k-means: cluster {} is empty'.format(c))
            else:
                SSE = np.append(SSE, sum_sq_error(X, k, N, C, idx))
                C[c] = my_mean(X[idx[:] == c], axis=0)

        # show cluster centres at iteration i
        print('[{}] Iteration: '.format(i+1), C.tolist())

        # If assignments were maintained, terminate
        if np.array_equal(idx, idx_prev):
            return C, SSE

        # Store current assignment to check if it changes in next iteration
        idx_prev = idx

    return C, idx, SSE


def sq_dist(U, v):
    return np.sum((U-v)**2, axis=1)


def sum_sq_error(A, K, N, centres, idx):

    z = np.zeros((K, N), dtype=bool)
    error = 0

    for row in range(K):
        for col in range(N):
            if idx[col] == row:
                z[row, col] = True

    for cluster in range(K):
        error += np.sum(sq_dist(A[z[cluster, :]], centres[cluster]))

    error *= (1 / N)

    return error
