import numpy as np


def sq_dist(U, v):
    """ Square distance between matrix and vector or 2 vectors """
    return np.sum((U-v)**2, axis=1)


def vec_sq_dist(X, Y):
    """ Vectorised square distance matrix between two matrices X and Y:
    d_{ij} is the distance between X[i] and Y[i] """

    N = len(X)
    M = len(Y)

    # YY = np.diag(np.dot(Y, Y.T))
    # XX = np.diag(np.dot(X, X.T))
    # this doesn't work because the dot product creates a matrix that is too big for memory
    XX = np.zeros(N)
    for i in range(N):
        XX[i] = np.dot(X[i], X[i])

    YY = np.zeros(M)
    for i in range(M):
        YY[i] = np.dot(Y[i], Y[i])

    DI = np.tile(XX, (M, 1)).T - 2 * np.dot(X, Y.T) + np.tile(YY, (N, 1))

    return DI
