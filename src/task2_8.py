from run_mgcs import *
from time import time
from comp_confmat import *
import scipy.io as sio


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





