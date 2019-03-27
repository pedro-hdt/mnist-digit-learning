from montage import *
from load_my_data_set import *
from my_dist import *
from my_kMeansClustering import *
from task1_1 import *
from task1_2 import *
from task1_3 import *
from task1_4 import *
from task1_5 import *
from task1_6 import *
from task1_7 import *
from task1_8 import *
import sys
import getopt
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans
from time import time


def run_all(visual):
    run_task1_1(visual)
    run_task1_2(visual)
    run_task1_3(visual)
    run_task1_4(visual)
    run_vec_dist(visual)
    run_k_means(visual)
    run_task1_5(visual)


def run_task1_1(visual):
    plt.clf()
    task1_1(Xtrn, Ytrn)
    if visual:
        plt.show()


def run_task1_2(visual):
    plt.clf()
    M = task1_2(Xtrn, Ytrn)
    plt.savefig(fname='../results/task1_2_imgs.pdf')
    sio.savemat(file_name='../results/task1_2_M.mat', mdict={'M': M})
    if visual:
        plt.show()


def run_task1_3(visual):
    plt.clf()
    EVecs, EVals, CumVar, MinDims = task1_3(Xtrn)
    if visual:
        plt.show()
    sio.savemat(file_name='../results/task1_3_evecs.mat', mdict={'EVecs': EVecs})
    sio.savemat(file_name='../results/task1_3_evals.mat', mdict={'EVals': EVals})
    sio.savemat(file_name='../results/task1_3_cumvar.mat', mdict={'CumVar': CumVar})
    sio.savemat(file_name='../results/task1_3_mindims.mat', mdict={'MinDims': MinDims})


def run_task1_4(visual):
    plt.clf()
    EVecs, EVals, CumVar, MinDims = task1_3(Xtrn)
    plt.clf()
    task1_4(EVecs)
    plt.suptitle('First 10 Principal Components')
    plt.savefig(fname='../results/task1_4_imgs.pdf')
    if visual:
        plt.show()


def run_vec_dist():
    X = Xtrn
    Y = Xtrn[:10]
    my_DI = vec_sq_dist(X, Y)
    DI = cdist(X, Y, 'sqeuclidean')
    if np.allclose(my_DI, DI):
        print 'Your DI matches the SciPy implementation!'
        diff = np.abs(my_DI - DI)
        print 'Total error: ' + str(np.sum(diff))
        print 'Maximum error: ' + str(np.max(diff))


def run_k_means(visual):

    my_C, idx, SSE = my_kMeansClustering(Xtrn, 10, Xtrn[:10])
    print 'Clusters assigned: ' + str(idx)
    print 'Data labels: ' + str(Ytrn)
    print SSE

    C, distortion = kmeans(Xtrn, k_or_guess=Xtrn[:10], iter=500)
    if np.allclose(my_C, C):
        print 'Your k-means matches the SciPy implementation!'
        diff = np.abs(my_C - C)
        print 'Total error: ' + str(np.sum(diff))
        print 'Maximum error: ' + str(np.max(diff))

    if visual:
        montage(C)
        plt.suptitle('Library function')
        plt.show()
        montage(my_C)
        plt.suptitle('Your result'.format())
        plt.show()


def run_task1_5(visual, cached=False):

    Ks = np.array([1, 2, 3, 4, 5, 7, 10, 15, 20])

    if not cached:
        start_time = time()
        task1_5(Xtrn, Ks)
        print 'Elapsed time: {} secs'.format(time() - start_time)

    for k in Ks:
        # load data
        my_C = sio.loadmat(file_name='../results/task1_5_c_{}.mat'.format(k))['C']

        # check against std lib implementation
        C, distortion = kmeans(Xtrn, k_or_guess=Xtrn[:k], iter=500)

        print '-------------------------- k = {} -------------------------'.format(k)
        diff = np.abs(my_C - C)
        if np.allclose(my_C, C):
            print 'Your k-means matches the SciPy implementation!'
        else:
            print 'Your k-means does not match the SciPy implementation... :\'('
        print 'Total error: ' + str(np.sum(diff))
        print 'Maximum error: ' + str(np.max(diff))

        if visual:
            # visualise
            montage(C)
            plt.suptitle('Library function')
            plt.show()
            montage(my_C)
            plt.suptitle('Your result')
            plt.show()


def run1_6(visual):
    pass


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'fsvc')
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    visual = False
    cached = False

    if len(opts) == 0:
        run_all()

    # print opts
    # print args

    for o, a in opts:
        if o == '-v':
            visual = True
        elif o == '-c':
            cached = True
        elif o == '-f':
            globals()['run_task' + args[0]](visual)
        elif o == '-s':
            for task in args:
                if task == 'k':
                    run_k_means(visual)
                elif task == 'dist':
                    run_vec_dist(visual)
                else:
                    task_n = task[0]
                    subtask_n = task[2:]
                    func_name = 'run_task' + task_n + '_' + subtask_n
                    print 'Running task ' + task
                    if task == '1.5':
                        globals()[func_name](visual, cached)
                    else:
                        globals()[func_name](visual)
        else:
            assert False, "Invalid option!"


def usage():
    print 'Usage: \"python2.7 run_task1\" followed by: '
    print '-c (cached) to avoid recomputing values when they are saved in .mat files'
    print '-v (visual) to open the plots as the task runs'
    print '-f (full) for the full task followed by the number'
    print 'e.g. for task 1: '
    print '\"python2.7 run_cw -f 1\"'
    print '-s (specific) for specific sub-tasks followed by the number'
    print 'e.g. for task 1.1: '
    print '\"python2.7 run_cw -s 1.1\"'


if __name__ == '__main__':

    # Import data
    Xtrn, Ytrn, Xtst, Ytst = load_my_data_set('../data')

    # Normalise
    Xtrn = Xtrn / 255.0
    Xtst = Xtst / 255.0

    # Run
    main()
