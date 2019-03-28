import scipy.io as sio


def task1_7(MAT_ClusterCentres, MAT_M, MAT_evecs, MAT_evals, posVec, nbins):
    """
    Write a Matlab function that visualises the cross section of cluster regions with a 2D-PCA
    plane, where the position of the plane is specified by a point vector. The plotting range
    should be m +- 5sigma, where m is the mean and sigma is the standard deviation of the data on
    the corresponding PCA axis.
    Input:
    :param MAT_ClusterCentres: filename of cluster centre matrix
    :param MAT_M             : MAT filename of mean vectors of (K+1)-by-D, where K is
           the number of classes (which is 10 for the MNIST data)
    :param MAT_evecs         : filename of eigenvector matrix of D-by-D
    :param MAT_evals         : filename of eigenvalue vector of D-by-1
    :param posVec            : 1-by-D vector (double) to specify the position of the plane
    :param nbins             : scalar (integer) to specify the number of bins for each PCA axis

    Output:
    :return: Dmap : nbins-by-nbins matrix (uint8) - each element represents
            the cluster number that the point belongs to.
    """

    Dmap = []

    C = sio.loadmat(file_name=MAT_ClusterCentres)['C']
    M = sio.loadmat(file_name=MAT_M)['M']
    EVecs = sio.loadmat(file_name=MAT_evecs)['EVecs']
    EVals = sio.loadmat(file_name=MAT_evals)['EVals']



    return Dmap
