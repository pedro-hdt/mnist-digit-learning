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

    Dmap = np.zeros((nbins, nbins))
    pass

    return Dmap
