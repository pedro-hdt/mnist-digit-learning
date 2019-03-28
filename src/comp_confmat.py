import numpy as np


def comp_confmat(Ytrues, Ypreds, K):
    """
    Write a Python function that creates a confusion matrix
    :param Ytrues: N-by-1 ground truth label vector
    :param Ypreds: N-by-1 predicted label vector
    :param K: number of classes
    :returns:
        CM : K-by-K confusion matrix, where CM(i,j) is
        the number of samples whose target is the ith class
        that was classified as j
        acc : accuracy (i.e. correct classification rate)
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