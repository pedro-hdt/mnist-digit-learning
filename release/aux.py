import numpy as np
from time import clock
from scipy.stats import mode
from logdet import *


def my_mean(x):
    """
    Returns the mean vector over the rows of x or the mean if x is a column vector

    :param x: ndarray for which the mean is to be calculated
    :type x: numpy.ndarray
    :return: mean vector of dimensions 1-by-D where D is x.shape[1]
    """
    return (1.0 * np.sum(x, axis=0)) / len(x)


def sq_dist(U, v):
    """
    Square distance between matrix and vector or 2 vectors

    :param U: Matrix
    :param v: vector or matric of the same size as U
    """
    return np.sum((U-v)**2, axis=1)


def vec_sq_dist(X, Y):
    """
    Vectorised square distance matrix between two matrices X and Y:
    d_{ij} is the distance between X[i] and Y[j]

    :param X: matrix of points
    :param Y: matrix of points
    :return: DI: DI_{ij} is the distance between X[i] and Y[j]
    """

    N = len(X)
    M = len(Y)

    # YY = np.diag(np.dot(Y, Y.T))
    # XX = np.diag(np.dot(X, X.T))
    # this doesn't work because the dot product creates a matrix that is too big for memory
    # For clarification on what's happening here, check the Gaussian classification function
    XX = np.einsum('ij,ji->i', X, X.T)
    if Y.ndim == 1:
        return sq_dist(X, Y)
    YY = np.einsum('ij,ji->i', Y, Y.T)

    # again, np.dot here causes a memory error in task 2.2 but so does np.tile,
    # meaning my laptop cannot handle the DI matrix itself so we restrict the dataset size
    DI = np.tile(XX, (M, 1)).T + np.tile(YY, (N, 1)) - 2 * np.dot(X, Y.T)

    return DI


def my_kMeansClustering(X, k, initialCentres, maxIter=500):
    """
    Write a Python function that carries out the k-means clustering and returns

    :param X: N-by-D matrix (double) of input sample data
    :param k: scalar (integer) - the number of clusters
    :param initialCentres: k-by-D matrix (double) of initial cluster centres
    :param maxIter: scalar (integer) - the maximum number of iterations
    :returns: tuple (C, idx, SSE)

    - C - k-by-D matrix (double) of cluster centres
    - idx - N-by-1 vector (integer) of cluster index table
    - SSE - (L+1)-by-1 vector (double) of sum-squared-errors where L is the number of iterations done
    """

    N = len(X)

    idx = np.zeros((N, 1))
    idx_prev = np.zeros((N, 1))
    C = initialCentres

    # initialise error list
    SSE = np.zeros((maxIter+1, 1))

    # Compute Squared Euclidean distance (i.e. the squared distance)
    # between each cluster centre and each observation
    for i in range(maxIter):

        dist = vec_sq_dist(X, C)

        # Assign data to clusters
        # Ds are the actual distances and idx are the cluster assignments
        idx = dist.argmin(axis=1).T  # find min dist. for each observation

        # add error to the list
        SSE[i] = sum_sq_error(dist, k, idx, N)

        # Update cluster centres
        for cluster in range(k):
            # check the number of samples assigned to this cluster
            if np.sum(idx == cluster) == 0:
                print('k-means: cluster {} is empty'.format(cluster))
            else:
                class_k = X[idx[:] == cluster]
                C[cluster] = my_mean(class_k)

        # If assignments were maintained, terminate
        if np.array_equal(idx, idx_prev):
            # but first add final error to the list
            SSE[i+1] = sum_sq_error(dist, k, idx, N)
            return C, idx, SSE[:i+2]

        # Store current assignment to check if it changes in next iteration
        idx_prev = idx

    # add final error to the list
    SSE[-1] = sum_sq_error(dist, k, idx, N)

    return C, idx, SSE


def sum_sq_error(dist, k, idx, N):
    """
    Computes the sum squared error during k means clustering

    :param dist: distance matrix
    :param k: number of cluster centres
    :param idx: cluster assignments
    :param N: number of samples
    :return: Sum squared error
    """
    error = 0

    for cluster in range(k):
        error += np.sum(dist[idx[:] == cluster, cluster])

    error *= (1. / N)

    return error


def comp_pca(X):
    """
    Write a Python function that computes the principal components of a data set
    The eigenvalues should be sorted in descending order, so that lambda_1 is the largest and lambda_D is
    the smallest, and i'th column of EVecs should hold the eigenvector that corresponds to lambda_i

    :param X: N-by-D matrix (double)
    :return: tuple (EVecs, EVals)
    :rtype: (numpy.ndarray, numpy.ndarray)

    - Evecs: D-by-D matrix (double) contains all eigenvectors as columns.
      NB: follow the Task 1.3 specifications on eigenvectors.
    - EVals: Eigenvalues in descending order, D x 1 vector
      (Note that the i-th columns of Evecs should corresponds to the i-th element in EVals)
    """

    N = len(X)
    D = X.shape[1]

    # Mean shift the original matrix
    X_shift = X - my_mean(X)

    # Covariance matrix
    # note we use 1.0, otherwise it gets cast to int and results in all 0s
    covar_m = 1.0 / (N - 1) * np.dot(X_shift.T, X_shift)

    # Find the eigenvectors and eigenvalues
    # EVecs will be the principal componentsbout that
    # library function returns unit vectors so we need nto worry a
    EVals, EVecs = np.linalg.eigh(covar_m)

    # The first element of each eigenvector must be non-negative
    for i in range(EVecs.shape[1]):
        if EVecs[0, i] < 0:
            EVecs[:, i] *= -1

    # Order eigenvalues in descending order
    # Note the slicing notation gives us a view rather than a copy of the array
    # This is good for efficiency!
    idx = EVals.argsort()[::-1]
    EVals = EVals[idx]

    # Order eigenvectors by the same order
    EVecs = EVecs[:, idx]

    # We need to force evals to a col vector in 2D to meet the spec
    return EVecs, EVals.reshape((D, 1))


def run_knn_classifier(Xtrn, Ytrn, Xtst, Ks):
    """
    Write a Python function for classification with k-NN

    :param Xtrn: M-by-D training data matrix
    :param Ytrn: M-by-1 label vector for Xtrn
    :param Xtst: N-by-D test data matrix
    :param Ks: L-by-1 vector of the numbers of nearest neighbours in Xtrn
    :return: Ypreds : N-by-L matrix of predicted labels for Xtst
    """

    N = len(Xtst)
    L = len(Ks)
    Ypreds = np.zeros((N, L), dtype='uint8')

    start_time = clock()

    # Compute distance from every point in test set to every point in dataset
    dist_mat = vec_sq_dist(Xtrn, Xtst)

    # get sorting index for all distances
    idx = dist_mat.argsort(axis=0)

    overhead_runtime = clock() - start_time
    print 'Overhead runtime: ', overhead_runtime

    for l in range(L):

        start_time = clock()

        k = Ks[l] # k in k-NN

        for n in range(N):

            # extract list of closest training samples using the first k entries of the index we calculated
            k_nn_labels = Ytrn[idx[:k, n]]

            # we predict it is the most common value (ie the mode) of the k-nn
            Ypreds[n, l], _ = mode(k_nn_labels)

        runtime = clock() - start_time + overhead_runtime
        print '\nRuntime of k-NN for k = {}'.format(Ks[l])
        print runtime

    return Ypreds


def my_gaussian_classify(Xtrn, Ytrn, Xtst, epsilon):
    """
    Write a Python function for the classification with a single Gaussian distribution per class
    where Xtrn, Ytrn, Xtst, and Ypreds are the same as those in Task1. epsilon is a scalar
    (double) for the regularisation of covariance matrix described in Lecture 8, in which we add
    a small positive number (epsilon) to the diagonal elements of covariance matrix,
    i.e. SIGMA <-- SIGMA + (epsilon)I, where I is the identity matrix.

    Note that, as opposed to Tasks 2.3 and 2.4 that use PCA, we do not apply PCA to the
    data in this task.

    :param Xtrn: M-by-D training data matrix
    :param Ytrn: M-by-1 label vector for Xtrn
    :param Xtst: N-by-D test data matrix
    :param epsilon:
    :return: tuple (Cpreds, Ms, Covs)

    - Cpreds: N-by-1 matrix of predicted labels for Xtst
    - Ms: K-by-D matrix of mean vectors where Ms[k, :] is the sample mean vector for class k.
    - Covs: K-by-D-by-D 3D array of covariance matrices, where Cov[k, :, :] is the covariance
      matrix (after the regularisation) for class k.
    """

    # Number of classes
    K = 10

    # Size of the matrices
    N = len(Xtst)
    M = len(Xtrn)
    D = len(Xtrn[0])

    Cpreds = np.zeros((N, 1))
    Ms = np.zeros((K, D))
    Covs = np.zeros((K, D, D))
    inv_Covs = np.zeros((K, D, D))
    priors = np.zeros(K)

    # Bayes classification with multivariate Gaussian distributions
    for C_k in range(K):

        # Extract class samples
        class_samples = Xtrn[Ytrn[:] == C_k]

        # Calculate prior probabilities
        priors[C_k] = (1.0 * len(class_samples)) / M

        # Estimate mean
        Ms[C_k] = my_mean(class_samples)

        # Estimate covariance matrix
        X_shift = class_samples - Ms[C_k]
        Covs[C_k] = (1.0 / len(class_samples)) * np.dot(X_shift.T, X_shift) + epsilon * np.identity(D)
        inv_Covs[C_k] = np.linalg.inv(Covs[C_k])

    # For each class, calculate the log posterior probability for all samples
    log_post_probs = np.zeros((K, N))
    for C_k in range(K):

        # Extract the mean and mean shift the data
        mu = Ms[C_k]
        Xtst_shift = Xtst - mu

        # Formula derived from (with extra vectorization) lect note 9 (equation 9.9 on page 4)
        # https://www.inf.ed.ac.uk/teaching/courses/inf2b/learnnotes/inf2b-learn09-notes-nup.pdf
        # for clarification on the einstein summation see:
        # https://stackoverflow.com/questions/14758283/is-there-a-numpy-scipy-dot-product-calculating-only-the-diagonal-entries-of-the
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html
        log_post_probs[C_k] = \
            - 0.5 * np.einsum('ij,jk,ki->i', Xtst_shift, inv_Covs[C_k], Xtst_shift.T) \
            - 0.5 * logdet(Covs[C_k]) \
            + np.log(priors[C_k])

    # Finally, assign to each sample the class that maximises the log posterior probaility
    Cpreds = log_post_probs.argmax(axis=0).astype('uint8')

    return Cpreds, Ms, Covs


def run_mgcs(Xtrn, Ytrn, Xtst, epsilon, L):
    """
    Write a Python function that applies k-means clustering to each class
    to obtain multiple Gaussian classifiers per class, and carries out classification.

    :param Xtrn: M-by-D training data matrix (double)
    :param Ytrn: M-by-1 label vector for Xtrain (uint8)
    :param Xtst: N-by-D test data matrix (double)
    :param epsilon: A scalar parameter for regularisation (double)
    :param L: scalar (integer) of the number of Gaussian distributions per class
    :return: triple of (Ypreds, MMs, MCovs)

    - Ypreds : N-by-1 matrix of predicted labels for Xtest (integer)
    - MMs     : (L*K)-by-D matrix of mean vectors (double)
    - MCovs   : (L*K)-by-D-by-D 3D array of covariance matrices (double)
    """

    # Number of classes
    K = 10

    N = len(Xtst)
    D = Xtst.shape[1]

    # Return vars
    MMs = np.zeros((L * K, D))
    MCovs = np.zeros((L * K, D, D))

    # Internal vars
    inv_Covs = np.zeros((L * K, D, D))
    priors = np.zeros(L * K)
    log_post_probs = np.zeros((L * K, N))

    # Clustering and training
    for C_k in range(K):

        # Extract class samples
        class_samples = Xtrn[Ytrn[:] == C_k]

        # Choose L random initial cluster centres and perform k-means
        centres = class_samples[np.random.randint(0, len(class_samples), L)]
        _, idx, _ = my_kMeansClustering(class_samples, L, centres)

        # Gaussian estimation (training)
        for subclass in range(L):

            # Extract class samples
            subclass_samples = class_samples[idx[:] == subclass]

            # Calculate prior probabilities
            priors[C_k*L+subclass] = (1.0 * len(subclass_samples)) / N

            # Estimate mean
            MMs[C_k*L+subclass] = my_mean(subclass_samples)

            # Estimate covariance matrix (and its inverse)
            X_shift = subclass_samples - MMs[C_k*L+subclass]
            MCovs[C_k*L+subclass] = (1.0 / len(subclass_samples)) * np.dot(X_shift.T, X_shift) + epsilon * np.identity(D)
            inv_Covs[C_k*L+subclass] = np.linalg.inv(MCovs[C_k*L+subclass])

    # Classification
    for C_k in range(K):

        for subclass in range(L):

            # Extract the mean and mean shift the data
            mu = MMs[C_k*L+subclass]
            Xtst_shift = Xtst - mu

            # Formula derived from (with extra vectorization) lect note 9 (equation 9.9 on page 4)
            # https://www.inf.ed.ac.uk/teaching/courses/inf2b/learnnotes/inf2b-learn09-notes-nup.pdf
            # for clarification on the einstein summation see:
            # https://stackoverflow.com/questions/14758283/is-there-a-numpy-scipy-dot-product-calculating-only-the-diagonal-entries-of-the
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html
            log_post_probs[C_k*L+subclass] = \
                - 0.5 * np.einsum('ij,jk,ki->i', Xtst_shift, inv_Covs[C_k*L+subclass], Xtst_shift.T) \
                - 0.5 * logdet(MCovs[C_k*L+subclass]) \
                + np.log(priors[C_k*L+subclass])

    # Assign to each sample the class that maximises the log posterior probaility
    Ypreds = log_post_probs.argmax(axis=0)

    # Finally, because we have L gaussians per class, we divide the assignments by L
    # and take the ceiling which gives us the corresponding final class.
    # (Multiplication by 1.0 ensures that result is not cast to an int)
    Ypreds = (Ypreds / (1.0 * L)).astype('uint8')

    return Ypreds, MMs, MCovs


def comp_confmat(Ytrues, Ypreds, K):
    """
    Write a Python function that creates a confusion matrix

    :param Ytrues: N-by-1 ground truth label vector
    :param Ypreds: N-by-1 predicted label vector
    :param K: number of classes
    :returns: tuple (CM, acc)

        - CM : K-by-K confusion matrix, where CM(i,j) is
          the number of samples whose target is the ith class
          that was classified as j
        - acc : accuracy (i.e. correct classification rate)
    """

    # Initialise the matrix with 0s so we can increment
    CM = np.zeros((K, K), dtype='int32')
    n = len(Ypreds)

    # For each classifcation we increment the corresponding entry
    # in the confusion matrix by 1
    for y in range(n):
        i = Ytrues[y]
        j = Ypreds[y]
        CM[i, j] += 1

    # Sum of the diagonal of the confusion matrix is the number of
    # correct predictions, form which we compute the accuracy
    correct_preds = CM.trace() # trace returns sum of diagonal
    acc = (1.* correct_preds) / n

    return CM, acc
