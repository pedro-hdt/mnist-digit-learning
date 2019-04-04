import numpy as np


def sq_dist(U, v):
    """ Square distance between matrix and vector or 2 vectors """
    return np.sum((U-v)**2, axis=1)


def vec_sq_dist(X, Y):
    """ Vectorised square distance matrix between two matrices X and Y:
    d_{ij} is the distance between X[i] and Y[j] """

    N = len(X)
    M = len(Y)

    # YY = np.diag(np.dot(Y, Y.T))
    # XX = np.diag(np.dot(X, X.T))
    # this doesn't work because the dot product creates a matrix that is too big for memory
    # For clarification on what's happening here, check the Gaussian classification function
    XX = np.einsum('ij,ji->i', X, X.T)
    YY = np.einsum('ij,ji->i', Y, Y.T)

    # again, np.dot here causes a memory error in task 2.2 but so does np.tile,
    # meaning my laptop cannot handle the DI matrix itself so we restrict the dataset size
    DI = np.tile(XX, (M, 1)).T + np.tile(YY, (N, 1)) - 2 * np.dot(X, Y.T)

    return DI
