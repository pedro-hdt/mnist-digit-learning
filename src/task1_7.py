import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


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

    Dmap = np.zeros((nbins, nbins), dtype='uint8')

    # Load all the data
    C = sio.loadmat(file_name=MAT_ClusterCentres)['C']
    M = sio.loadmat(file_name=MAT_M)['M']
    EVecs = sio.loadmat(file_name=MAT_evecs)['EVecs']
    EVals = sio.loadmat(file_name=MAT_evals)['EVals']

    K = len(M) - 1

    # Principal components in 2D are 2 first eigenvectors (w/ highest eigenvalues)
    PC = EVecs[:, :2]

    # Transform the original data C to the principal subspace
    projected_C = np.dot(C, PC)
    sigma = EVals[:2]

    for i in range(K):

        # extract relevant mean vector
        mean = M[i]

        # Plot the data in the new basis
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(projected_C[:, 0], projected_C[:, 1], marker='.')
        ax.xlabel('1st Principal Component');
        ax.ylabel('2nd Principal Component');
        ax.box(on=True)
        ax.set_xlim((mean-5*sigma[0], mean+5*sigma[1]))
        ax.set_ylim((mean-5*sigma[0], mean+5*sigma[1]))


    return Dmap
