from comp_pca import *
from my_mean import *


def task2_3(X, Y):
    """
    Write a Python function that does the following:
        1. Transform X to the data in the 2D space spanned by the first two principal components.
        2. Estimate the parameters (mean vector and covariance matrix) of Gaussian distribution
        for each class in the 2D space.
        3. On a single graph, plot a contour of the distribution for each class using plot() function.
        Do not use functions for plotting contours such as contour(). The lengths of longest
        / shortest axis of an ellipse should be proportional to the standard deviation for the
        axis. (Please note that contours of different distributions plotted by this method
        do not necessary give the set of points of the same pdf value.)
    :param X: M-by-D data matrix (double)
    :param Y: M-by-1 label vector (uint8)
    """

    # These parameters control the number of classes and pca dimensions respectively
    # They are irrelevant for the assignment, and could be hardcoded but this
    # makes the code reusable
    n_classes = 10
    pca_dim = 2

    # 1. Projecting the data into the 2D principal subspace
    EVecs, EVals = comp_pca(X)
    X_pc = np.dot(X, EVecs)[:, :pca_dim]

    # 2. Estimating the parameters of Gaussian for each class
    mean = np.zeros((n_classes, 2))
    covar_m = np.zeros((n_classes, pca_dim, pca_dim))
    invcovar_m = np.zeros((n_classes, pca_dim, pca_dim))
    std_dv = np.zeros((n_classes, 2))

    for C_k in range(n_classes):
        class_samples = X_pc[Y[:] == C_k]
        mean[C_k] = my_mean(class_samples)
        X_pcshift = class_samples - mean[C_k]
        covar_m[C_k] = (1.0 / len(class_samples)) * np.dot(X_pcshift.T, X_pcshift)
        invcovar_m[C_k] = np.linalg.inv(covar_m[C_k])
        std_dv[C_k] = np.diag(covar_m[C_k])
        # In FAQ page:
        # Q: Which type of covariance matrix should I use - the one normalised
        #    by N or N-1?
        # A: Please use the one normalised by N, because MLE is assumed.

    # 3. On a single graph plot a contour of the distribution for each class
    # (without using the contour function)
    for C_k in range(n_classes):
        x1, x2 = comp_pca(covar_m[C_k])


    print 'hello'
