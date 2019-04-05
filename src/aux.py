import numpy as np


def my_mean(x):
    """
    Returns the mean vector over the rows of x or the mean if x is a column vector

    :param x: ndarray for which the mean is to be calculated
    :type x: numpy.ndarray
    :return: mean vector of dimensions 1-by-D where D is x.shape[1]
    """
    return (1.0 * np.sum(x, axis=0)) / len(x)


def sq_dist(U, v):
    """
    Square distance between matrix and vector or 2 vectors

    :param U: Matrix
    :param v: vector or matric of the same size as U
    """
    return np.sum((U-v)**2, axis=1)


def vec_sq_dist(X, Y):
    """
    Vectorised square distance matrix between two matrices X and Y:
    d_{ij} is the distance between X[i] and Y[j]

    :param X: matrix of points
    :param Y: matrix of points
    :return: DI: DI_{ij} is the distance between X[i] and Y[j]
    """

    N = len(X)
    M = len(Y)

    # YY = np.diag(np.dot(Y, Y.T))
    # XX = np.diag(np.dot(X, X.T))
    # this doesn't work because the dot product creates a matrix that is too big for memory
    # For clarification on what's happening here, check the Gaussian classification function
    XX = np.einsum('ij,ji->i', X, X.T)
    if Y.ndim == 1:
        return sq_dist(X, Y)
    YY = np.einsum('ij,ji->i', Y, Y.T)

    # again, np.dot here causes a memory error in task 2.2 but so does np.tile,
    # meaning my laptop cannot handle the DI matrix itself so we restrict the dataset size
    DI = np.tile(XX, (M, 1)).T + np.tile(YY, (N, 1)) - 2 * np.dot(X, Y.T)

    return DI


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
    """
    Computes the sum squared error during k means clustering

    :param dist: distance matrix
    :param k: number of cluster centres
    :param idx: cluster assignments
    :param N: number of samples
    :return: Sum squared error
    """
    error = 0

    for cluster in range(k):
        error += np.sum(dist[idx[:] == cluster, cluster])

    error *= (1. / N)

    return error


def comp_pca(X):
    """
    Write a Python function that computes the principal components of a data set
    The eigenvalues should be sorted in descending order, so that lambda_1 is the largest and lambda_D is
    the smallest, and i'th column of EVecs should hold the eigenvector that corresponds to lambda_i

    :param X: N-by-D matrix (double)
    :return: tuple (EVecs, EVals)
    :rtype: (numpy.ndarray, numpy.ndarray)

    - Evecs: D-by-D matrix (double) contains all eigenvectors as columns.
      NB: follow the Task 1.3 specifications on eigenvectors.
    - EVals: Eigenvalues in descending order, D x 1 vector
      (Note that the i-th columns of Evecs should corresponds to the i-th element in EVals)
    """

    N = len(X)
    D = X.shape[1]

    # Mean shift the original matrix
    X_shift = X - my_mean(X)

    # Covariance matrix
    # note we use 1.0, otherwise it gets cast to int and results in all 0s
    covar_m = 1.0 / (N - 1) * np.dot(X_shift.T, X_shift)

    # Find the eigenvectors and eigenvalues
    # EVecs will be the principal componentsbout that
    # library function returns unit vectors so we need nto worry a
    EVals, EVecs = np.linalg.eig(covar_m)

    # The first element of each eigenvector must be non-negative
    for i in range(EVecs.shape[1]):
        if EVecs[0, i] < 0:
            EVecs[:, i] *= -1

    # Order eigenvalues in descending order
    # Note the slicing notation gives us a view rather than a copy of the array
    # This is good for efficiency!
    idx = EVals.argsort()[::-1]
    EVals = EVals[idx]

    # Order eigenvectors by the same order
    EVecs = EVecs[:, idx]

    # We need to force evals to a col vector in 2D to meet the spec
    return EVecs, EVals.reshape((D, 1))