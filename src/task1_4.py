from montage import *


def task1_4(EVecs):
    """
    Write a Python function that displays the images of the first ten principal axes of PCA
    using montage() so that all the images are shown in a single graph

    :param Evecs: the same format as in comp_pca.py
    """

    montage(EVecs[:, :10].T)
