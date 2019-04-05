import numpy as np
from my_mean import *


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
