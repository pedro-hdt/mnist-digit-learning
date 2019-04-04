from my_kMeansClustering import *
import matplotlib.pyplot as plt


def task1_8(X, Y, k):
    """
    This is a mini research project, in which you are asked to investigate the k-means clustering in
    terms of initial cluster centres, i.e. how different initial cluster centres result in different cluster
    centres, for which employ SSE to measure the clustering performance. Report your experiments
    and findings in your report.

    :param X: N-by-D ndarray (this is the dataset to perform clustering on)
    :param k: number of clusters
    """
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
        print '\nMethod: {} '.format(method)
        print 'Final error: ', np.asscalar(sse[-1])
        print 'Number of iterations: ', len(sse)-1

    plt.show()


