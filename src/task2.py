from time import time
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from aux import run_knn_classifier, comp_confmat, run_mgcs, my_gaussian_classify, my_mean, comp_pca


def task2_1(Xtrn, Ytrn, Xtst, Ytst, Ks):
    """
    Write the following Python function, that does the following:
    1. Runs a classification experiment on the data set using run_knn_classifier
    2. Measures the user time taken for the classification experiment, and display the time
    (in seconds) to the standard output (i.e. display).
    3. Saves the confusion matrix for each k to a matrix variable cm, and save it with the file
    name 'task2_1_cm{k}.mat', where k denotes the number of nearest neighbours as specified in Ks.
    4. Displays the following information (to the standard output):
        - k: The number of nearest neighbours
        - N: The number of test samples
        - Nerrs: The number of wrongly classified test samples
        - acc: Accuracy (i.e. correct classification rate)
    :param Xtrain: M-by-D training data matrix (double)
    :param Ytrain: M-by-1 label vector (unit8) for Xtrain
    :param Xtest: M-by-D test data matrix (double)
    :param Ytest: M-by-1 label vector (unit8) for Xtest
    :param Ks: 1-by-L vector (integer) of the numbers of nearest neighbours in Xtrain
    """

    # Numbers refer to tasks in docstring above

    start_time = time() # 2.
    Ypreds = run_knn_classifier(Xtrn, Ytrn, Xtst, Ks) # 1.
    runtime = time() - start_time # 2.
    print 'Elapsed time in k-nn: {}\n'.format(runtime) #2.

    N = len(Ytst)
    L = len(Ks)
    for l in range(L):

        # 3.
        k = Ks[l]
        CM, acc = comp_confmat(Ytst, Ypreds[:, l], 10)
        sio.savemat(file_name='task2_1_cm{}.mat'.format(k), mdict={'cm': CM})

        # 4.
        Nerrs = N - CM.trace()
        print '\nk = {}'.format(k)
        print 'N = {}'.format(N)
        print 'Nerrs = {}'.format(Nerrs)
        print 'acc = {}'.format(acc)


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
    :return: Dmap: nbins-by-nbins matrix (uint8) - each element repressents
    the cluster number that the point belongs to.
    """

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

    # Create grid with range mean+-5sigma on each axis as specified
    xbounds = [np.asscalar(projected_mean[:, 0] - (5 * sigma[0])), np.asscalar(projected_mean[:, 1] + 5 * (sigma[0]))]
    ybounds = [np.asscalar(projected_mean[:, 0] - (5 * sigma[1])), np.asscalar(projected_mean[:, 1] + 5 * (sigma[1]))]
    xrange = np.linspace(xbounds[0], xbounds[1], num=nbins)
    yrange = np.linspace(ybounds[0], ybounds[1], num=nbins)
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
    Dmap = Dmap.reshape((nbins, nbins))

    # Plot Dmap (normalised to fit the colormap)
    fig, ax = plt.subplots()
    ax.imshow(Dmap / (1.0 * k), cmap='viridis', origin='lower', extent=xbounds + ybounds)
    ax.set(xlabel='1st Principal Component',
           ylabel='2nd Principal Component',
           xlim=xbounds,
           ylim=ybounds)
    fig.canvas.set_window_title('Task 2.2')
    fig.suptitle('k-NN decision regions for k = {}'.format(k))

    return Dmap


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

        # This is alternative code for this section, as seen in
        # https://stackoverflow.com/questions/10952060/plot-ellipse-with-matplotlib-pyplot-python
        #
        # unit = np.array([1, 0])
        # angle = acos( np.dot(vs[0], unit) / (np.linalg.norm(unit) * np.linalg.norm(vs[0])) )
        # rot_mat = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
        # rot_contour = np.dot(rot_mat, contour)
        # plt.plot(rot_contour[0], rot_contour[1])

    # Prettify figure
    ax.set(xlabel='1st Principal Component',
           ylabel='2nd Principal Component')
    fig.canvas.set_window_title('Task 2.3')
    fig.suptitle('Contours of distributions for each class')


def task2_4(X, Y):
    """
    Write a Python function that calculates correlation r12 on 2D-PCA for each class and for
    all the classes (i.e. whole data)

    :param X: M-by-D data matrix (double)
    :param Y: M-by-1 label vector (unit8) for X
    :return: Corrs  : (K+1)-by-1 vector (double) of correlation r_{12}
             for each class k = 1,...,K, and the last element holds the correlation
             for the whole data, i.e. Xtrain.
    """

    # These parameters control the number of classes and pca dimensions respectively
    # They are 'irrelevant' for the assignment, and could be hardcoded but this
    # makes the code reusable and more readable
    n_classes = 10
    pca_dim = 2

    Corrs = np.zeros((n_classes+1, 1))

    # Projecting the data into the 2D principal subspace
    EVecs, EVals = comp_pca(X)
    X_pc = np.dot(X, EVecs)[:, :pca_dim]

    # Calculating the correlation r_12 for each class
    for C_k in range(n_classes):
        class_samples = X_pc[Y[:] == C_k]
        mean = my_mean(class_samples)
        X_pcshift = class_samples - mean
        covar_m = (1.0 / len(class_samples)) * np.dot(X_pcshift.T, X_pcshift)
        Corrs[C_k] = covar_m[0, 1] / (np.diag(covar_m).prod())**0.5

    # Calculating the correlation r_12 for the whole data
    mean = my_mean(X_pc)
    X_pcshift = X_pc - mean
    covar_m = (1.0 / len(X_pc)) * np.dot(X_pcshift.T, X_pcshift)
    Corrs[10] = covar_m[0, 1] / (np.diag(covar_m).prod())**0.5

    return Corrs


def task2_5(Xtrn, Ytrn, Xtst, Ytst, epsilon):
    """
    Write a Python function for a classification experiment that does the following:
    1. Calls the classification function with epsilon=0.01.
    2. Measures the user time taken for the classification experiment, and displays it
    (in seconds) to the standard output.
    3. Obtains the confusion matrix, stores it to a matrix variable cm, and saves it with the
    file name 'task2_5_cm.mat'.
    4. Copy the mean vector and covariance matrix for Class 10, i.e., Ms[10,:] and Covs[10,:,:],
    to new variables, M10 and Cov10, respectively, in the following manner:
    M10 = Ms[10,:]
    Cov10 = Covs[10,:,:]
    and save them as 'task2_5_m10.mat' and 'task2_5_cov10.mat', respectively.
    5 Displays the following information (to the standard output).
        - N The number of test samples
        - Nerrs The number of wrongly classified test samples
        - acc Accuracy (i.e. correct classification rate)

    :param Xtrn: M-by-D training data matrix (double)
    :param Ytrn: M-by-1 label vector (unit8) for Xtrain
    :param Xtst: M-by-D test data matrix (double)
    :param Ytst: M-by-1 label vector (unit8) for Xtest
    :param epsilon: a scalar variable (double) for covariance regularisation
    """

    # Number tags refer to the subtasks as numbered in the docstring above

    start_time = time() # 2.
    Ypreds, Ms, Covs = my_gaussian_classify(Xtrn, Ytrn, Xtst, epsilon) # 1.
    print 'Elapsed time in Gaussian classification: {} secs'.format(time() - start_time) #2.

    # 3.
    cm, acc = comp_confmat(Ytst, Ypreds, 10)
    sio.savemat(file_name='task2_5_cm.mat', mdict={'cm': cm})

    # 4.
    M10 = Ms[9]     # class 10 has index 9
    Cov10 = Covs[9] # class 10 has index 9
    sio.savemat(file_name='task2_5_m10.mat', mdict={'M10': M10})
    sio.savemat(file_name='task2_5_cov10.mat', mdict={'Cov10': Cov10})

    # 5.
    N = len(Xtst)
    Nerrs = N - cm.trace()
    print 'N = {}'.format(N)
    print 'Nerrs = {}'.format(Nerrs)
    print 'acc = {}'.format(acc)


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


def task2_7(Xtrn, Ytrn, Xtst, Ytst, epsilon, ratio):
    """
    This task aims to investigate the effect of amount of training data on classification performance
    for Gaussian classifiers.
    Write a Python function that runs an experiment using a subset of training data.
    where ratio specifies the ratio of training data to use. If it is 0.9, use the first 90% of
    samples in Xtrain

    :param Xtrn: M-by-D training data matrix (double)
    :param Ytrn: M-by-1 label vector (unit8) for Xtrain
    :param Xtst: M-by-D test data matrix (double)
    :param Ytst: M-by-1 label vector (unit8) for Xtest
    :param epsilon: a scalar variable (double) for covariance regularisation
    :param ratio: scalar (double) - ratio of training data to use.
    :return:
        CM : K-by-K matrix(integer) of confusion matrix
        acc : scalar (double) of correct classification rate
    """

    # Extract relevant part of dataset
    data_limit = int(ratio * len(Ytrn))

    # Classify
    Ypreds, _, _ = my_gaussian_classify(Xtrn[:data_limit], Ytrn[:data_limit], Xtst, epsilon)

    # Analyse results
    CM, acc = comp_confmat(Ytst, Ypreds, 10)

    return CM, acc


def task2_8(Xtrn, Ytrn, Xtst, Ytst, epsilon, L):
    """
    Write a Python function that:

    1. Calls the classification function for multiple gaussians per class
    2. Measures the user time taken for the classification experiment, and display the time (in seconds)
       to the standard output.
    3. Obtains the confusion matrix, stores it to a matrix variable cm, and saves it with the file name
       'task2_8_cm_L.mat'.
    4. Copies the mean vectors and covariance matrices for Class 1, to new variables, Ms1 and Covs1, respectively,
       in the following manner: ``Ms1 = MMs[1:L,:]`` and  ``Covs1 = MCovs[1:L,:,:]``
    5. Saves Ms1 and Covs1 as 'task2_8_gL_m1.mat' and 'task2_8_gL_cov1.mat', respectively, where L is the value of L
    6. Displays the following information (to the standard output):

       a. N: The number of test samples
       b. Nerrs: The number of wrongly classified test samples
       c. acc: Accuracy (i.e. correct classification rate)

    :param Xtrn: M-by-D training data matrix (double)
    :param Ytrn: M-by-1 label vector (unit8) for Xtrain
    :param Xtst: M-by-D test data matrix (double)
    :param Ytst: M-by-1 label vector (unit8) for Xtest
    :param epsilon: a scalar variable (double) for covariance regularisation
    :param L: scalar (integer) of the number of Gaussian distributions per class
    """

    # Number tags refer to steps as in the docstring above

    start_time = time() # 2.
    Ypreds, MMs, MCovs = run_mgcs(Xtrn, Ytrn, Xtst, epsilon, L) # 1.
    print 'Elapsed time in MGC: {} secs'.format(time() - start_time) # 2.

    # 3.
    cm, acc = comp_confmat(Ytst, Ypreds, 10)
    sio.savemat(file_name='task2_8_cm_{}.mat'.format(L), mdict={'cm': cm})

    # 4.
    Ms1 = MMs[1:L, :]
    Covs1 = MCovs[1:L, :, :]

    # 5.
    sio.savemat(file_name='task2_8_g{}_m1.mat'.format(L), mdict={'Ms1': Ms1})
    sio.savemat(file_name='task2_8_g{}_cov1.mat'.format(L), mdict={'Covs1': Covs1})

    # 6.
    N = len(Xtst)
    Nerrs = N - cm.trace()
    print 'N = {}'.format(N)
    print 'Nerrs = {}'.format(Nerrs)
    print 'acc = {}'.format(acc)