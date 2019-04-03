from run_gaussian_classifiers import *
from comp_confmat import *


def task2_7(Xtrn, Ytrn, Xtst, Ytst, epsilon, ratio):
    """
    This task aims to investigate the effect of amount of training data on classification performance
    for Gaussian classifiers.
    Write a Python function that runs an experiment using a subset of training data.
    where ratio specifies the ratio of training data to use. If it is 0.9, use the first 90% of
    samples in Xtrain
    :param Xtrain: M-by-D training data matrix (double)
    :param Ytrain: M-by-1 label vector (unit8) for Xtrain
    :param Xtest: M-by-D test data matrix (double)
    :param Ytest: M-by-1 label vector (unit8) for Xtest
    :param epsilon: a scalar variable (double) for covariance regularisation
    :param ratio: scalar (double) - ratio of training data to use.
    :return:
        CM : K-by-K matrix(integer) of confusion matrix
        acc : scalar (double) of correct classification rate
    """

    data_limit = int(ratio * len(Ytrn))
    Ypreds, _, _ = my_gaussian_classify(Xtrn[:data_limit], Ytrn[:data_limit], Xtst, epsilon)
    CM, acc = comp_confmat(Ytst, Ypreds, 10)

    return CM, acc
