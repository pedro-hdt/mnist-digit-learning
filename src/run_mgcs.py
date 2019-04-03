from my_kMeansClustering import *
from my_mean import *
from logdet import *


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