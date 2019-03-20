import numpy as np
import matplotlib.pyplot as plt


def task1_1(X, Y):
    """Write a Python function that displays the images of the first ten samples for each class
    using montage() function, so that each figure shows the ten samples for class Ck, where
    k = 1, . . . , 10.
    Inputs:
    X: M-by-D data matrix (of doubles) where M is the number of samples, and D is the the number of 
    elements in a sample. Note that each sample is represented as a row vector rather than a column vector.
    Y : M-by-1 label vector (uint8) for X. Y(i) is the class number of X(i,:)."""

    # for each of the 10 classes
    for i in range(10):

        # Initialise a figure with 2 by 5 subplots and prettify it
        f, axarr = plt.subplots(2, 5)
        f.canvas.set_window_title('Class {}'.format(i))
        f.subplots_adjust(wspace=0, hspace=0.2)
        f.suptitle('First ten smaples of digit {}'.format(i), size=15)
        f.set_size_inches(7.45, 3)


        # for each of the first 10 samples of digit i
        samples = 0
        j = 0
        while samples in range(10):
            x = samples / 5
            y = samples % 5
            if Y[j] == i:
                axarr[x][y].imshow(np.reshape(X[j, :], (28, 28)), cmap='gray')
                axarr[x][y].axis('off')
                samples += 1
            j += 1

        plt.savefig(fname='../results/task1_1_imgs_class{}.pdf'.format(i)) # TODO: Before submitting comment this
        # plt.show() TODO: Before submitting uncomment this
