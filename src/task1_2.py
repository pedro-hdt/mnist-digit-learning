import numpy as np
import scipy.io as sio
from montage import *
from my_mean import *


def task1_2(X, Y):
    """
    Write a Python function that calculates a mean vector of data for each class (k = 1, . . . , K,
    where K = 10) and for all, and displays the images of K + 1 mean vectors in a single graph
    using montage() function.

    :param X: M-by-D data matrix (of doubles) where M is the number of samples, and D is the the number of
    elements in a sample. Note that each sample is represented as a row vector rather than a column vector.
    :param Y: M-by-1 label vector (uint8) for X. Y(i) is the class number of X[i, :].
    :return: M: (K+1)-by-D mean vector matrix (float64), where K (not an input) is the number of classes, and D
    is the same as in Task 1.1. M[K+1,:] is the mean vector of the whole data.
    """

    # Extract dimensions
    D = X.shape[1]

    # Initialise return matrix
    M = np.zeros((11, D))

    # for each of the 10 classes
    for C_k in range(10):

        # Extract all samples of the class
        class_samples = X[Y[:] == C_k]
        # Get the mean on those samples
        M[C_k] = my_mean(class_samples)

    # Calculate overall mean of the dataset
    M[10] = my_mean(X)

    # Plot
    montage(M)
    fig = plt.gcf()
    fig.suptitle('Mean vectors for each class and overall dataset', size=13)
    fig.canvas.set_window_title('Task 1.2')
    plt.show()

    return M



