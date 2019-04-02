import numpy as np
import scipy.io as sio
from my_mean import *
from run_knn_classifier import *
import matplotlib.pyplot as plt

def task2_2(X, Y, k, MAT_evecs, MAT_evals, posVec, nbins):
    """
    Write a Python function that visualises the cross section of decision regions of k-NN with
    a 2D-PCA plane, where the position of the plane is specified by a point vector. Use the
    same specifications for plotting as in Task 1.7
    :param X: M-by-D data matrix (double)
    :param k: scalar (integer) - the number of nearest neighbours
    :param MAT_evecs: MAT filename of eigenvector matrix of D-by-D
    :param MAT_evals: MAT filename of eigenvalue vector of D-by-1
    :param posVec: 1-by-D vector (double) to specity the position of the plane
    :param nbins: scalar (integer) - the number of bins for each PCA axis
    :return: Dmap  : nbins-by-nbins matrix (uint8) - each element repressents
    the cluster number that the point belongs to.
    """

    D = posVec.shape[1]
    Dmap = np.zeros((1, nbins * nbins), dtype='uint8')

    # Load the data
    EVecs = sio.loadmat(file_name=MAT_evecs)['EVecs']
    EVals = sio.loadmat(file_name=MAT_evals, squeeze_me=True)['EVals']

    # Extract relevant mean vector and transform it to the principal subspace
    mean = my_mean(X)
    projected_posVec = np.dot(posVec, EVecs)
    projected_mean = np.dot(mean, EVecs) - projected_posVec

    # Extract std deviation (sqrt(var))
    sigma = EVals[:2] ** 0.5

    # Create grid
    xrange = np.linspace(projected_mean[:, 0] - (5 * sigma[0]), projected_mean[:, 1] + 5 * (sigma[0]), num=nbins)
    yrange = np.linspace(projected_mean[:, 0] - (5 * sigma[1]), projected_mean[:, 1] + 5 * (sigma[1]), num=nbins)
    xx_pc, yy_pc = np.meshgrid(xrange, yrange)

    # Padding the grid with 0's to make it match the dimensions od the unprojected data
    grid_pc = np.zeros((D, nbins * nbins))
    grid_pc[0] = np.array([xx_pc]).ravel()
    grid_pc[1] = np.array([yy_pc]).ravel()

    # 'Unproject' grid according to the specifications at
    # http://www.inf.ed.ac.uk/teaching/courses/inf2b/coursework/inf2b_cwk2_2019_notes_task1_7.pdf
    # y is projected data
    # V is EVecs
    # x is unprojected data
    # p is position vector
    V_inv = np.linalg.inv(EVecs.T)
    grid = np.dot(V_inv, grid_pc) + posVec.T

    # Classify the points in the grid
    Dmap = run_knn_classifier(X, Y, grid.T, [k])

    # Create a color map for plotting
    colormap = plt.get_cmap(lut=10)
    colors = colormap(np.arange(10))

    # Plot the data in the new basis
    plt.scatter(xx_pc, yy_pc, c=colors[Dmap.ravel()])

    plt.xlabel('1st Principal Component')
    plt.ylabel('2nd Principal Component')
    plt.box(on=True)
    plt.xlim(projected_mean[:, 0] - (5 * sigma[0]), projected_mean[:, 1] + 5 * (sigma[0]))
    plt.ylim(projected_mean[:, 0] - (5 * sigma[1]), projected_mean[:, 1] + 5 * (sigma[1]))
    plt.suptitle('k-NN decision regions for k = {}'.format(k))
    # plt.show() # TODO uncomment this before submission

    return Dmap
