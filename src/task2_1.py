from run_knn_classifier import *
from comp_confmat import *
from time import time
import scipy.io as sio


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
    print 'Elapsed time in k-nn: {}\n'.format(time() - start_time) #2.

    N = len(Ytst)
    L = len(Ks)
    for l in range(L):

        # 3.
        k = Ks[l]
        CM, acc = comp_confmat(Ytst, Ypreds[:, l], 10)
        sio.savemat(file_name='task2_1_cm{}.mat'.format(k), mdict={'cm': CM})

        # 4.
        Nerrs = N - CM.trace()
        print 'k = {}'.format(k)
        print 'N = {}'.format(N)
        print 'Nerrs = {}'.format(Nerrs)
        print 'acc = {}'.format(acc)
