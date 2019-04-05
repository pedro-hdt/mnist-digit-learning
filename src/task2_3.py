from comp_pca import *
from my_mean import *
import matplotlib.pyplot as plt
from math import acos, cos, sin , pi


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
    # makes the code easier to adapt to other situations
    n_classes = 10
    pca_dim = 2

    # 1. Projecting the data into the 2D principal subspace
    EVecs, EVals = comp_pca(X)
    X_pc = np.dot(X, EVecs)[:, :pca_dim]

    # 2. Estimating the parameters of Gaussian for each class
    mean = np.zeros((n_classes, pca_dim))
    covar_m = np.zeros((n_classes, pca_dim, pca_dim))
    std_dv = np.zeros((n_classes, pca_dim))

    # Initialise our figure
    fig, ax = plt.subplots()

    for C_k in range(n_classes):

        class_samples = X_pc[Y[:] == C_k]
        mean[C_k] = my_mean(class_samples)
        X_pcshift = class_samples - mean[C_k]
        covar_m[C_k] = (1.0 / len(class_samples)) * np.dot(X_pcshift.T, X_pcshift)
        std_dv[C_k] = np.sqrt(np.diag(covar_m[C_k]))
        # In FAQ page:
        # Q: Which type of covariance matrix should I use - the one normalised
        #    by N or N-1?
        # A: Please use the one normalised by N, because MLE is assumed.

        # 3. On a single graph plot a contour of the distribution for each class
        # (without using the contour function)

        # We do this based on slides 9-10 from
        # https://www.inf.ed.ac.uk/teaching/courses/inf2b/learnSlides/inf2b-learnlec09-full.pdf

        # Define contour size using the std deviation
        a = std_dv[C_k][0]
        b = std_dv[C_k][1]

        # Define a circle
        x = np.linspace(0, 2*np.pi, 100)
        contour = np.array([a * np.cos(x), b * np.sin(x)])

        # Use the eigenvectors of the covariance matrix to create a linear transformation
        # of the circle (adding the mean to translate the location)
        vecs, vals = comp_pca(covar_m[C_k])
        myrot = np.dot(vecs.T, contour) + mean[C_k].reshape([pca_dim, 1])
        ax.plot(myrot[0], myrot[1])
        ax.text(mean[C_k, 0], mean[C_k, 1], s=str(C_k))
        ax.set(xlabel='1st Principal Component',
               ylabel='2nd Principal Component')
        fig.canvas.set_window_title('Task 2.3')
        fig.suptitle('Contours of distributions for each class')

        # This is alternative code for this section, as seen in
        # https://stackoverflow.com/questions/10952060/plot-ellipse-with-matplotlib-pyplot-python
        #
        # unit = np.array([1, 0])
        # angle = acos( np.dot(vs[0], unit) / (np.linalg.norm(unit) * np.linalg.norm(vs[0])) )
        # rot_mat = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
        # rot_contour = np.dot(rot_mat, contour)
        # plt.plot(rot_contour[0], rot_contour[1])

    plt.show()
