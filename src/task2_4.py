import numpy as np
from comp_pca import *


def task2_4(X, Y):
    """
    Write a Python function that calculates correlation r12 on 2D-PCA for each class and for
    all the classes (i.e. whole data)
    :param Xtrain: M-by-D data matrix (double)
    :param Ytrain: M-by-1 label vector (unit8) for X
    :return: Corrs  : (K+1)-by-1 vector (double) of correlation r_{12}
    for each class k = 1,...,K, and the last element holds the correlation
    for the whole data, i.e. Xtrain.
    """

    # These parameters control the number of classes and pca dimensions respectively
    # They are irrelevant for the assignment, and could be hardcoded but this
    # makes the code reusable
    n_classes = 10
    pca_dim = 2

    Corrs = np.zeros((n_classes+1, 1))

    # Projecting the data into the 2D principal subspace
    EVecs, EVals = comp_pca(X)
    X_pc = np.dot(X, EVecs)[:, :pca_dim]

    # Calculating the correlation r_12 for each class
    for C_k in range(n_classes):
        class_samples = X_pc[Y[:] == C_k]
        mean = my_mean(class_samples)
        X_pcshift = class_samples - mean
        covar_m = (1.0 / len(class_samples)) * np.dot(X_pcshift.T, X_pcshift)
        Corrs[C_k] = covar_m[0, 1] / (np.diag(covar_m).prod())**0.5

    # Calculating the correlation r_12 for the whole data
    mean = my_mean(X_pc)
    X_pcshift = X_pc - mean
    covar_m = (1.0 / len(X_pc)) * np.dot(X_pcshift.T, X_pcshift)
    Corrs[10] = covar_m[0, 1] / (np.diag(covar_m).prod())**0.5

    return Corrs
