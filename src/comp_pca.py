import numpy as np
import numpy.linalg as la


def comp_pca(X):
    # Input:
    # X: N * D matrix (double)
    # Output:
    # Evecs: D-by-D matrix (double) contains all eigenvectors as columns
    # NB: follow the Task 1.3 specifications on eigenvectors.
    # EVals: Eigenvalues in descending order, D x 1 vector
    # (Note that the i-th columns of Evecs should corresponds to the i-th element in EVals)

    N = len(X)

    # Mean shift the original matrix
    X_shift = X - X.mean(axis=0)

    # Covariance matrix
    # note we use 1.0, otherwise it gets cast to int and results in all 0s
    covar_m = 1.0 / (N - 1) * np.dot(X_shift.T, X_shift)

    # find the eigenvectors and eigenvalues
    # EVecs will be the principal components
    EVals, EVecs = la.eig(covar_m)
    # TODO: this might return complex values! do we need to deal with that?

    # The first element of each eigenvector must be non-negative
    for i in range(len(EVecs[0])):
        if EVecs[0, i] < 0:
            print 'negating'
            EVecs[:, i] *= -1

    # Order eigenvalues
    idx = EVals.argsort()[::-1]
    EVals = EVals[idx]


    # Order eigenvectors by the same order
    EVecs = EVecs[:, idx]

    return EVecs, EVals
