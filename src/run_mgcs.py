from my_kMeansClustering import *
from run_gaussian_classifiers import my_gaussian_classify


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
    Ypreds = np.zeros((N, 1))
    MMs =  np.zeros((L * K, D))
    MCovs = np.zeros((L * K, D, D))

    for C_k in range(K):

        # Extract class samples
        class_samples = Xtrn[Ytrn[:] == C_k]

        # Choose L random initial cluster centres and perform k-means
        centres = class_samples[np.random.randint(0, len(class_samples), L)]
        C, idx, SSE = my_kMeansClustering(class_samples, L, centres)

        for subclass in range(L):

            subclass_samples = class_samples[idx[:] == subclass]
            Ypred, Ms, Covs = my_gaussian_classify()






    pass

    return Ypreds, MMs, MCovs