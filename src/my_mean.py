import numpy as np


def my_mean(x):
    return np.sum(x, axis=0) / len(x)
