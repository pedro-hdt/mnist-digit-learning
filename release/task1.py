import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from aux import my_mean, comp_pca, my_kMeansClustering, vec_sq_dist
from montage import *
from time import clock


def task1_1(X, Y):
    """
    Write a Python function that displays the images of the first ten samples for each class
    using montage() function, so that each figure shows the ten samples for class Ck, where
    k = 1, . . . , 10.

    :param X: M-by-D data matrix (of doubles) where M is the number of samples, and D is the the number of
    elements in a sample. Note that each sample is represented as a row vector rather than a column vector.
    :param Y: M-by-1 label vector (uint8) for X. Y(i) is the class number of X[i, :].
    """

    print '\n==================== Running task 1.1 ===================='

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
        fig.suptitle('First ten smaples of digit {}'.format(C_k), size=13)
        fig.canvas.set_window_title('Task 1.1 - Class {}'.format(C_k))

        # Saving the files in both pdf and png for the report
        plt.savefig(fname='../results/task1_1_imgs_class{}.pdf'.format(C_k))
        plt.savefig(fname='../results/task1_1_imgs_class{}.png'.format(C_k))


def task1_2(X, Y):
    """
    Write a Python function that calculates a mean vector of data for each class (k = 1, . . . , K,
    where K = 10) and for all, and displays the images of K + 1 mean vectors in a single graph
    using montage() function.

    :param X: M-by-D data matrix (of doubles) where M is the number of samples, and D is the the number of
    elements in a sample. Note that each sample is represented as a row vector rather than a column vector.
    :param Y: M-by-1 label vector (uint8) for X. Y(i) is the class number of X[i, :].
    :return: M: (K+1)-by-D mean vector matrix (float64), where K (not an input) is the number of classes, and D
    is the same as in Task 1.1. M[K+1,:] is the mean vector of the whole data.
    """

    print '\n==================== Running task 1.2 ===================='

    # Extract dimensions
    D = X.shape[1]

    # Initialise return matrix
    M = np.zeros((11, D))

    # for each of the 10 classes
    for C_k in range(10):

        # Extract all samples of the class
        class_samples = X[Y[:] == C_k]
        # Get the mean on those samples
        M[C_k] = my_mean(class_samples)

    # Calculate overall mean of the dataset
    M[10] = my_mean(X)

    # Plot
    montage(M)
    fig = plt.gcf()
    fig.suptitle('Mean vectors for each class and overall dataset', size=13)
    fig.canvas.set_window_title('Task 1.2')

    return M


def task1_3(X):
    """
    Write a Python function task1_3() that
        #. carries out PCA (i.e. calls comp_pca),
        #. computes cumulative variances
        #. plots the cumulative variances
        #. finds the minimum number of PCA dimensions to cover
           70%, 80%, 90%, 95% of the total variance

    :param X: M-by-D data matrix (double)
    :type X: numpy.ndarray
    :return: tuple (EVecs, EVals, CumVar, MinDims)
    :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)

    - EVecs, Evals: same as in comp_pca.py
    - CumVar  : D-by-1 vector (double) of cumulative variance
    - MinDims : 4-by-1 vector (integer) of the minimum number of PCA dimensions to cover
      70%, 80%, 90%, and 95% of the total variance.
    """

    print '\n==================== Running task 1.3 ===================='

    D = X.shape[1]

    # Carry out PCA
    EVecs, EVals = comp_pca(X)

    # Compute cumulative variances
    # Eigenvectors are variances, so we do a cumulative sum of them
    CumVar = np.cumsum(EVals, dtype='float64').reshape((D, 1))

    # Plot cumulative variances
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(CumVar))+1, CumVar)
    ax.set(xlabel='# of principal components', ylabel='Cumulative variance')
    fig.suptitle('Cumulative Variance', size=13)
    fig.canvas.set_window_title('Task 1.3')
    fig.savefig('../results/task1_3_graph.pdf')
    fig.savefig('../results/task1_3_graph.png')

    # Find minimum number of PCA dimensions to cover percentages of variance
    MinDims = np.zeros((4, 1), dtype='int32')

    # Total variance is just the sum of all variances (last element of the cumulative sum)
    TotalVar = CumVar[-1]

    # Compute the ratios of total variance that the cumulative variances cover
    CumRatios = CumVar / TotalVar

    # Find minimum number of dimensions to cover each of the ratios/percentages
    ratio = [0.7, 0.8, 0.9, 0.95]
    j = 0  # our index into the ratio array
    for i in range(len(CumRatios)):  # we go through the ratios
        # once we find one that is at least what we are looking for
        if CumRatios[i] >= ratio[j]:
            MinDims[j] = i + 1 # we record its index + 1(the number of dimensions)
            j += 1          # and move on to look for the next ratio
        # once we have the 4 values we need, terminate the loop
        if j == 4:
            break

    return EVecs, EVals, CumVar, MinDims


def task1_4(EVecs):
    """
    Write a Python function that displays the images of the first ten principal axes of PCA
    using montage() so that all the images are shown in a single graph

    :param Evecs: the same format as in comp_pca.py
    """

    print '\n==================== Running task 1.4 ===================='

    montage(EVecs[:, :10].T)


def task1_5(X, Ks):
    """
    Write a Python function that calls the k-means clustering function for each k in a vector Ks,
    using the first k samples in the data set X as the initial cluster centres, and saves the returned
    C, idx, and SSE as 'task1_5_c_{k}.mat', 'task1_5_idx_{k}.mat', and 'task1_5_sse_{k}.mat', respectively.

    :param X: M-by-D data matrix (double)
    :param Ks: 1-by-L vector (integer) of the numbers of nearest neighbours
    """

    print '\n==================== Running task 1.5 ===================='

    Cs = []
    idxs = []
    SSEs = []

    # Do all computation because saving is asynchronous which makes it dangerous
    for k in Ks:
        start_time = clock()
        C, idx, SSE = my_kMeansClustering(X, k, X[:k])
        runtime = clock() - start_time
        print 'Elapsed time in clustering for k = {}: {} secs'.format(k, runtime)
        Cs.append(np.copy(C))
        idxs.append(np.copy(idx))
        SSEs.append(np.copy(SSE))

    # Save all files
    for i in range(len(Ks)):
        k = Ks[i]
        sio.savemat(file_name='task1_5_c_{}.mat'.format(k), mdict={'C': Cs[i]})
        sio.savemat(file_name='task1_5_idx_{}.mat'.format(k), mdict={'idx': idxs[i]})
        sio.savemat(file_name='task1_5_sse_{}.mat'.format(k), mdict={'SSE': SSEs[i]})


def task1_6(MAT_ClusterCentres):
    """
    Write a Pyhton function that displays the image of each cluster centre, where you should use
    montage() function to put all the images into a single figure.

    Input:
    :param MAT_ClusterCentres: file name of the file that contains cluster centres C.
    """

    print '\n==================== Running task 1.6 ===================='

    C = sio.loadmat(file_name=MAT_ClusterCentres)['C']
    montage(C)


def task1_7(MAT_ClusterCentres, MAT_M, MAT_evecs, MAT_evals, posVec, nbins):
    """
    Write a Pyhton function that visualises the cross section of cluster regions with a 2D-PCA
    plane, where the position of the plane is specified by a point vector. The plotting range
    should be m +- 5sigma, where m is the mean and sigma is the standard deviation of the data on
    the corresponding PCA axis.

    :param MAT_ClusterCentres: filename of cluster centre matrix
    :param MAT_M: MAT filename of mean vectors of (K+1)-by-D, where K is
    the number of classes (which is 10 for the MNIST data)
    :param MAT_evecs: filename of eigenvector matrix of D-by-D
    :param MAT_evals: filename of eigenvalue vector of D-by-1
    :param posVec: 1-by-D vector (double) to specify the position of the plane
    :param nbins: scalar (integer) to specify the number of bins for each PCA axis

    :return: Dmap : nbins-by-nbins matrix (uint8) - each element represents
             the cluster number that the point belongs to.
    """

    print '\n==================== Running task 1.7 ===================='

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
    xbounds = [np.asscalar(projected_mean[:, 0] - (5 * sigma[0])), np.asscalar(projected_mean[:, 1] + 5 * (sigma[0]))]
    ybounds = [np.asscalar(projected_mean[:, 0] - (5 * sigma[1])), np.asscalar(projected_mean[:, 1] + 5 * (sigma[1]))]
    xrange = np.linspace(xbounds[0], xbounds[1], num=nbins)
    yrange = np.linspace(ybounds[0], ybounds[1], num=nbins)
    xx_pc, yy_pc = np.meshgrid(xrange, yrange)

    # Padding the grid with 0's to make it match the dimensions of the 'unprojected' data
    grid_pc = np.zeros((D, nbins * nbins))
    grid_pc[0] = np.array([xx_pc]).ravel()
    grid_pc[1] = np.array([yy_pc]).ravel()

    # 'Unproject' grid according to the specifications from
    # http://www.inf.ed.ac.uk/teaching/courses/inf2b/coursework/inf2b_cwk2_2019_notes_task1_7.pdf
    # y is the projected data
    # V is EVecs
    # x is the unprojected data
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

    # Plot Dmap (normalised to fit the colormap)
    fig, ax = plt.subplots()
    ax.imshow(Dmap / (1.0 * K), cmap='viridis', origin='lower', extent=xbounds + ybounds)
    ax.set(xlabel='1st Principal Component',
           ylabel='2nd Principal Component',
           xlim=xbounds,
           ylim=ybounds)
    fig.canvas.set_window_title('Task 1.7')
    fig.suptitle('Decision regions after k-means clustering for k = {}'.format(K))

    return Dmap



def task1_8(X, Y, k):
    """
    This is a mini research project, in which you are asked to investigate the k-means clustering in
    terms of initial cluster centres, i.e. how different initial cluster centres result in different cluster
    centres, for which employ SSE to measure the clustering performance. Report your experiments
    and findings in your report.

    :param X: N-by-D ndarray (this is the dataset to perform clustering on)
    :param k: number of clusters
    """

    print '\n==================== Running task 1.8 ===================='

    #==================================================================================#
    # Our 'control' is going to be the results for when we use the first ten samples
    # in the dataset as suggested previously.
    # To compare with that we will choose initial centres in the following ways:
    #   * Randomly pick 10 samples, from any class, from the dataset
    #   * Calculate the mean of the dataset and choose the 10 samples which are further
    #     away from it
    #
    #   In a real world scenario this would not be possible, but for the purpose of
    #   experimenting it is so we will also:
    #
    #   * Randomly pick 10 samples, one from each class, from the dataset
    #   * Use the mean of each class as the initial cluster centres
    #==================================================================================#

    D = X.shape[1]
    SSEs = []
    methods = ['first 10 samples',
               'random samples',
               'furthest from mean',
               'random sample of each class',
               'mean of each class']

    # 'Control'
    _, _, SSE_init = my_kMeansClustering(X, k, X[:k])
    SSEs.append(np.copy(SSE_init))


    # Random samples
    rand_samples = np.random.randint(0, len(X), k)
    centres = X[rand_samples]
    _, _, SSE_rand = my_kMeansClustering(X, k, centres)
    SSEs.append(np.copy(SSE_rand))

    # Furthest from the mean
    mean = my_mean(X)
    DI = vec_sq_dist(X, mean)
    furthest = DI.argsort(axis=0)[-10:]
    centres = X[furthest]
    _, _, SSE_furt = my_kMeansClustering(X, k, centres)
    SSEs.append(np.copy(SSE_furt))

    #======================================================================#
    # Methods from here on are not possible in true unsupervised learning! #
    #======================================================================#

    # Random sample of each class
    centres = np.zeros((k, D))
    for C_k in range(k):
        class_samples = X[Y[:] == C_k]
        rand_samples_class = np.random.randint(0, len(class_samples), 1)
        centres[C_k] = class_samples[rand_samples_class]
    _, _, SSE_rand_class = my_kMeansClustering(X, k, centres)
    SSEs.append(np.copy(SSE_rand_class))

    # Mean of each class
    centres = np.zeros((k, D))
    for C_k in range(k):
        class_samples = X[Y[:] == C_k]
        centres[C_k] = my_mean(class_samples)
    _, _, SSE_mean_class = my_kMeansClustering(X, k, centres)
    SSEs.append(np.copy(SSE_mean_class))

    # Plotting everything
    for i in range(len(SSEs)):
        sse = SSEs[i]
        method = methods[i]
        fig, ax = plt.subplots()
        x = np.arange(len(sse))
        ax.plot(x, sse)
        ax.set(xlabel='Iteration number', ylabel='SSE')
        if k == 1:
            ax.set(xticks=x)
        fig.suptitle('SSE for k = {}\nMethod: {}'.format(k, method))
        fig.canvas.set_window_title('Task 1.8')
        plt.savefig('../results/task1_8_{}.pdf'.format(i))
        plt.savefig('../results/task1_8_{}.png'.format(i))
        print '\nMethod: {} '.format(method)
        print 'Final error: ', np.asscalar(sse[-1])
        print 'Number of iterations: ', len(sse)-1

    plt.show()
