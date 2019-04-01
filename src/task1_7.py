import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from my_dist import *


def task1_7(MAT_ClusterCentres, MAT_M, MAT_evecs, MAT_evals, posVec, nbins):
    """
    Write a Pyhton function that visualises the cross section of cluster regions with a 2D-PCA
    plane, where the position of the plane is specified by a point vector. The plotting range
    should be m +- 5sigma, where m is the mean and sigma is the standard deviation of the data on
    the corresponding PCA axis.

    Input:
    :param MAT_ClusterCentres: filename of cluster centre matrix
    :param MAT_M: MAT filename of mean vectors of (K+1)-by-D, where K is
    the number of classes (which is 10 for the MNIST data)
    :param MAT_evecs: filename of eigenvector matrix of D-by-D
    :param MAT_evals: filename of eigenvalue vector of D-by-1
    :param posVec: 1-by-D vector (double) to specify the position of the plane
    :param nbins: scalar (integer) to specify the number of bins for each PCA axis

    Output:
    :return: Dmap : nbins-by-nbins matrix (uint8) - each element represents
            the cluster number that the point belongs to.
    """

    D = posVec.shape[1]
    Dmap = np.zeros((1, nbins*nbins), dtype='uint8')

    # Load all the data
    C = sio.loadmat(file_name=MAT_ClusterCentres)['C']
    K = len(C)
    M = sio.loadmat(file_name=MAT_M)['M']
    EVecs = sio.loadmat(file_name=MAT_evecs)['EVecs']
    EVals = sio.loadmat(file_name=MAT_evals, squeeze_me=True)['EVals']

    #Extract std deviation is sqrt(var)
    sigma = EVals[:2]**0.5

    # Extract relevant mean vector and transform it to the principal subspace
    mean = M[-1]
    projected_posVec = np.dot(posVec, EVecs)
    projected_mean = np.dot(mean, EVecs) - projected_posVec

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

    # Classify the grid
    for i in range(nbins*nbins):
        cell = grid[:, i].reshape((1, D))
        DI = vec_sq_dist(C, cell)
        assignment = np.asscalar(DI.argmin(axis=0))
        Dmap[:, i] = assignment

    Dmap = Dmap.reshape((nbins, nbins))

    # Create a color map for plotting
    colormap = plt.get_cmap(lut=K)

    # Plot the data in the new basis
    # plt.scatter(xx_pc, yy_pc, c=colors[Dmap.ravel()])
    plt.contourf(xx_pc, yy_pc, Dmap, cmap=colormap)

    plt.xlabel('1st Principal Component')
    plt.ylabel('2nd Principal Component')
    plt.box(on=True)
    plt.xlim(projected_mean[:, 0] - (5 * sigma[0]), projected_mean[:, 1] + 5 * (sigma[0]))
    plt.ylim(projected_mean[:, 0] - (5 * sigma[1]), projected_mean[:, 1] + 5 * (sigma[1]))

    return Dmap.reshape((nbins, nbins))
