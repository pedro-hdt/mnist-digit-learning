import numpy as np


def my_mean(x):
    """
    Returns the mean vector over the rows of x or the mean if x is a column vector

    :param x: ndarray for which the mean is to be calculated
    :type x: numpy.ndarray
    :return: mean vector of dimensions 1-by-D where D is x.shape[1]
    """
    return (1.0 * np.sum(x, axis=0)) / len(x)
