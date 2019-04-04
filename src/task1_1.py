import numpy as np
import matplotlib.pyplot as plt
from montage import *


def task1_1(X, Y):
    """
    Write a Python function that displays the images of the first ten samples for each class
    using montage() function, so that each figure shows the ten samples for class Ck, where
    k = 1, . . . , 10.

    :param X: M-by-D data matrix (of doubles) where M is the number of samples, and D is the the number of
    elements in a sample. Note that each sample is represented as a row vector rather than a column vector.
    :param Y: M-by-1 label vector (uint8) for X. Y(i) is the class number of X[i, :].
    """

    # for each of the 10 classes
    for C_k in range(10):

        # Extract first 10 samples of the class
        class_samples = X[Y[:] == C_k][:10]

        # For the 1st class do not initialise a new figure (so we don't get a blank window)
        # otherwise we do want a new figure so we can have multiple windows open at once
        if C_k != 0:
            plt.figure()
        montage(class_samples)

        # Prettify our plots with titles
        fig = plt.gcf()
        fig.suptitle('First ten smaples of digit {}'.format(C_k), size=15)
        fig.canvas.set_window_title('Class {}'.format(C_k))

        # Saving the files in both pdf and png for the report
        plt.savefig(fname='../results/task1_1_imgs_class{}.pdf'.format(C_k))
        plt.savefig(fname='../results/task1_1_imgs_class{}.png'.format(C_k))

    plt.show()