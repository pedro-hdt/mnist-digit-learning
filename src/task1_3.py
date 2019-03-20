import numpy as np
from comp_pca import *

def task1_3(X):
    # Input:
    # X : M-by-D data matrix (double)
    # Output:
    # EVecs, Evals: same as in comp_pca.py
    # CumVar  : D-by-1 vector (double) of cumulative variance
    # MinDims : 4-by-1 vector (integer) of the minimum number of PCA dimensions
    #           to cover 70%, 80%, 90%, and 95% of the total variance.

    EVecs, EVals = comp_pca(X)
    CumVar = np.cumsum(EVals)
    MinDims = []


    return EVecs, EVals, CumVar, MinDims