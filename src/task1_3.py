import numpy as np
import matplotlib.pyplot as plt
from comp_pca import *


def task1_3(X):
    """
    Write a Python function task1_3() that
        * carries out PCA (i.e. calls comp_pca),
        * computes cumulative variances
        * plots the cumulative variances
        * finds the minimum number of PCA dimensions to cover
        70%, 80%, 90%, 95% of the total variance

    Input:
    :param: X : M-by-D data matrix (double)

    Output:
    :return: EVecs, Evals: same as in comp_pca.py
    :return: CumVar  : D-by-1 vector (double) of cumulative variance
    :return: MinDims : 4-by-1 vector (integer) of the minimum number of PCA dimensions to cover
    70%, 80%, 90%, and 95% of the total variance.
    """

    # Carry out PCA
    EVecs, EVals = comp_pca(X)

    # Compute cumulative variances
    # Eigenvectors are variances, so we do a cumulative sum of them
    CumVar = np.cumsum(EVals, dtype='float64')

    # Plot cumulative variances
    plt.plot(np.arange(len(CumVar)), CumVar)
    plt.suptitle('Cumulative Variances', size=15)
    plt.savefig('../results/task1_3_graph.pdf')

    # Find minimum number of PCA dimensions to cover percentages of variance
    MinDims = np.zeros(4, dtype='int32')

    # Total variance is just the sum of all variances (last element of the cumulative sum)
    TotalVar = CumVar[-1]

    # Compute the ratios of total variance that the cumulative variances cover
    CumRatios = CumVar / TotalVar

    # Find minimum number of dimensions to cover each of the ratios/percentages
    ratio = [0.7, 0.8, 0.9, 0.95]
    j = 0  # our index into the ratio array
    for i in range(len(CumRatios)):  # we go through the ratios
        # once we find one that is at least what we are looking for
        if CumRatios[i] >= ratio[j]:
            MinDims[j] = i  # we record its index (the number of dimensions)
            j += 1          # and move on to look for the next ratio
        # once we have the 4 values we need, terminate the loop
        if j == 4:
            break

    return EVecs, EVals, CumVar, MinDims
