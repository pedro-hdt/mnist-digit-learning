import numpy as np
import scipy.io as sio
from my_mean import *
from run_gaussian_classifiers import my_gaussian_classify
import matplotlib.pyplot as plt


def task2_6(X, Y, epsilon, MAT_evecs, MAT_evals, posVec, nbins):
    """
    Write a Python function that visualises the cross section of decision
    regions of the Gaussian classifiers with a 2D-PCA plane.
    :param X: M-by-D data matrix (double)
    :param Y: M-by-1 label vector (uint8)
    :param epsilon: scalar (double) for covariance matrix regularisation
    :param MAT_evecs: MAT filename of eigenvector matrix of D-by-D
    :param MAT_evals: MAT filename of eigenvalue vector of D-by-1
    :param posVec: 1-by-D vector (double) to specity the position of the plane
    :param nbins: scalar (integer) - the number of bins for each PCA axis
    :return:
        Dmap: nbins-by-nbins matrix (uint8) - each element represents
        the cluster number that the point belongs to.
    """

    pca_dim = 2
    n_classes = 10

    D = posVec.shape[1]

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

    # Since the vectorisation in the gaussian classifier requires a large amount of memory,
    # we split the grid into chunks that can be independently classified, to avoid
    # a MemoryError.
    # We can do this since the classification of each sample is independent from all others
    # Note also that this will cause us to train the gaussian classifiers twice, but that
    # can be done in a couple of seconds so it is not a big problem

    n_chunks = 4
    chunk_size = grid.shape[1] / n_chunks
    Dmap = np.zeros(nbins * nbins, dtype='uint8')
    for n in range(n_chunks):
        low_limit = n * chunk_size
        up_limit = (n + 1) * chunk_size
        grid_chunk = grid[:, low_limit : up_limit].T
        print low_limit, up_limit
        Dmap[low_limit : up_limit], _, _ = my_gaussian_classify(X, Y, grid_chunk, epsilon)
    low_limit = n_chunks * chunk_size
    grid_chunk = grid[:, low_limit:].T
    print low_limit, 'onwards'
    Dmap[low_limit:], _, _ = my_gaussian_classify(X, Y, grid_chunk, epsilon)

    Dmap = Dmap.reshape((nbins, nbins))

    # Plot the data in the new basis
    # Create a color map for plotting
    colormap = plt.cm.viridis
    colors = [colormap(i) for i in np.linspace(0, 1, 20)]

    # Plot the data in the new basis
    #plt.scatter(xx_pc, yy_pc, c=colors[Dmap.ravel()])
    plt.figure()
    plt.contourf(xx_pc, yy_pc, Dmap, levels=range(10), colors=colors)
    # TODO choose plotting method after verifying

    plt.xlabel('1st Principal Component')
    plt.ylabel('2nd Principal Component')
    plt.box(on=True)
    plt.xlim(projected_mean[:, 0] - (5 * sigma[0]), projected_mean[:, 1] + 5 * (sigma[0]))
    plt.ylim(projected_mean[:, 0] - (5 * sigma[1]), projected_mean[:, 1] + 5 * (sigma[1]))
    plt.suptitle('Gaussian decision regions')

    return Dmap
