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
    sigma = EVals[:pca_dim] ** 0.5

    # Create grid
    xbounds = [np.asscalar(projected_mean[:, 0] - (5 * sigma[0])), np.asscalar(projected_mean[:, 1] + 5 * (sigma[0]))]
    ybounds = [np.asscalar(projected_mean[:, 0] - (5 * sigma[1])), np.asscalar(projected_mean[:, 1] + 5 * (sigma[1]))]
    xrange = np.linspace(xbounds[0], xbounds[1], num=nbins)
    yrange = np.linspace(ybounds[0], ybounds[1], num=nbins)
    xx_pc, yy_pc = np.meshgrid(xrange, yrange)

    # Padding the grid with 0's to make it match the dimensions of the unprojected data
    grid_pc = np.zeros((D, nbins * nbins))
    grid_pc[0] = xx_pc.ravel()
    grid_pc[1] = yy_pc.ravel()

    # 'Unproject' grid according to the specifications at
    # http://www.inf.ed.ac.uk/teaching/courses/inf2b/coursework/inf2b_cwk2_2019_notes_task1_7.pdf
    # y is projected data
    # V is EVecs
    # x is unprojected data
    # p is position vector
    V_inv = np.linalg.inv(EVecs.T)
    grid = np.dot(V_inv, grid_pc) + posVec.T

    # Previously, before introducing the einstein summation in the gaussian classification,
    # there would not be enough memory to classify the entire grid at once, so this solution was used:
    #
    # Since the vectorisation in the gaussian classifier requires a large amount of memory,
    # we split the grid into chunks that can be independently classified, to avoid
    # a MemoryError. Divide and conquer! =D
    # The value of n_chunks can be adjusted according to how much memory the machine has.
    # We can do this since the classification of each sample is independent from all others
    # Note also that this will cause us to train the gaussian classifiers twice, but that
    # can be done in a couple of seconds so it is not a big problem.
    # n_chunks = 1
    # chunk_size = grid.shape[1] / n_chunks
    # Dmap = np.zeros(nbins * nbins, dtype='uint8')
    # for n in range(n_chunks):
    #     low_limit = n * chunk_size
    #     up_limit = (n + 1) * chunk_size
    #     grid_chunk = grid[:, low_limit : up_limit].T
    #     print low_limit, up_limit
    #     Dmap[low_limit : up_limit], _, _ = my_gaussian_classify(X, Y, grid_chunk, epsilon)
    # low_limit = n_chunks * chunk_size
    # grid_chunk = grid[:, low_limit:].T
    # print low_limit, 'onwards'
    # Dmap[low_limit:], _, _ = my_gaussian_classify(X, Y, grid_chunk, epsilon)

    Dmap, _, _ = my_gaussian_classify(X, Y, grid.T, epsilon)
    Dmap = Dmap.reshape((nbins, nbins))

    # Plot Dmap (normalised to fit the colormap) in the new basis
    fig, ax = plt.subplots()
    ax.imshow(Dmap / (1.0 * n_classes), cmap='viridis', origin='lower', extent=xbounds + ybounds)
    ax.set(xlabel='1st Principal Component',
           ylabel='2nd Principal Component',
           xlim=xbounds,
           ylim=ybounds)
    fig.canvas.set_window_title('Task 2.6')
    fig.suptitle('Decision regions of Gaussian classifiers')

    return Dmap
