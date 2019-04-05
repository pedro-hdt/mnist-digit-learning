from run_gaussian_classifiers import *
from comp_confmat import *
from time import time
import scipy.io as sio
import sys


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

    sys.stdout = open('../results/task2_5_log.txt', 'w+')

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

    sys.stdout = sys.__stdout__
    with open('../results/task2_5_log.txt', 'r') as f:
        print f.read()
