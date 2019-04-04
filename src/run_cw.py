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
from task2_1 import *
from task2_2 import *
from task2_3 import *
from task2_4 import *
from task2_5 import *
from task2_6 import *
from task2_7 import *
from task2_8 import *
import sys
import os
import getopt
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans
from time import time
import hashlib


# TODO use clock instead of time


def run_all(visual):
    start_time = time()
    times = []
    t11 = run_task1_1(visual)
    times.append(t11)
    t12 = run_task1_2(visual)
    times.append(t12)
    t13 = run_task1_3(visual)
    times.append(t13)
    t14 = run_task1_4(visual)
    times.append(t14)

    run_vec_dist(visual)
    run_k_means(visual)

    t15 = run_task1_5(visual)
    times.append(t15)
    t16 = run_task1_6(visual)
    times.append(t16)
    print '\nAll tasks run successfully!'
    print 'Total runtime (including saving and plotting interruptions): {} secs'.format(time() - start_time)
    print 'Total runtime (excluding saving and plotting interruptions): {} secs'.format(sum(times))


def run_task1_1(visual):
    plt.clf()
    start_time = time()
    task1_1(Xtrn, Ytrn)
    runtime = time() - start_time
    return runtime

def run_task1_2(visual):
    plt.clf()
    start_time = time()
    M = task1_2(Xtrn, Ytrn)
    runtime = time() - start_time
    plt.savefig(fname='../results/task1_2_imgs.pdf')
    plt.savefig(fname='../results/task1_2_imgs.png')
    sio.savemat(file_name='../results/task1_2_M.mat', mdict={'M': M})
    if visual:
        plt.show()
    return runtime


def run_task1_3(visual):
    plt.clf()
    start_time = time()
    EVecs, EVals, CumVar, MinDims = task1_3(Xtrn)
    runtime = time() - start_time
    if visual:
        plt.show()
    sio.savemat(file_name='../results/task1_3_evecs.mat', mdict={'EVecs': EVecs})
    sio.savemat(file_name='../results/task1_3_evals.mat', mdict={'EVals': EVals})
    sio.savemat(file_name='../results/task1_3_cumvar.mat', mdict={'CumVar': CumVar})
    sio.savemat(file_name='../results/task1_3_mindims.mat', mdict={'MinDims': MinDims})
    return runtime


def run_task1_4(visual):
    plt.clf()
    EVecs, EVals, CumVar, MinDims = task1_3(Xtrn)
    plt.clf()
    start_time = time()
    task1_4(EVecs)
    runtime = time() - start_time
    plt.suptitle('First 10 Principal Components')
    plt.savefig(fname='../results/task1_4_imgs.pdf')
    plt.savefig(fname='../results/task1_2_imgs.png')
    if visual:
        plt.show()
        return runtime


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


def run_k_means(visual, k=10, test=False):

    if test:
        return verify_task1_5()

    my_C, idx, SSE = my_kMeansClustering(Xtrn, k, Xtrn[:k])
    print 'Clusters assigned: ' + str(idx)
    print 'Data labels: ' + str(Ytrn)

    # Write to file and reload to verify this is done correctly
    sio.savemat(file_name='../results/task1_5_c_{}.mat'.format(k), mdict={'C': my_C})

    if visual:
        test_k_means(k)


def test_k_means(k):

    print '\n---------------------------- k = {} ---------------------------'.format(k)
    my_C = sio.loadmat(file_name='../results/task1_5_c_{}.mat'.format(k))['C']
    my_distortion = sio.loadmat(file_name='../results/task1_5_sse_{}.mat'.format(k))['SSE'][0][-1] ** 0.5
    C, distortion = kmeans(Xtrn, k_or_guess=Xtrn[:k], iter=1)
    diff = np.abs(my_C - C)
    total_diff = np.sum(diff)
    if total_diff < 1:
        print 'Your k-means matches the SciPy implementation!'
    else:
        print 'Your k-means does not match the SciPy implementation... :\'('
    print 'Total diff: ' + str(total_diff)
    print 'Maximum diff: ' + str(np.max(diff))
    print 'Final distortion (you): ' + str(my_distortion)
    print 'Final distortion (lib): ' + str(distortion)

    # visualise
    montage(C)
    plt.suptitle('Library function')
    plt.show()
    montage(my_C)
    plt.suptitle('Your result')
    plt.show()


def verify_task1_5():

    Ks = Ks_clustering

    for k in Ks:
        test_k_means(k)


def run_task1_5(visual, cached=False):

    Ks = Ks_clustering

    if cached:
        verify_task1_5()
        return

    start_time = time()
    task1_5(Xtrn, Ks)
    runtime = time() - start_time
    print 'Elapsed time: {} secs'.format(runtime)

    # Move files to results directory to avoid cluttering our src folder
    for k in Ks:

        C, distortion = kmeans(Xtrn, k_or_guess=Xtrn[:k], iter=1)
        sio.savemat(file_name='../results/task1_5_c_{}_lib.mat'.format(k), mdict={'C': C, 'distortion': distortion})

        os.rename('task1_5_c_{}.mat'.format(k), '../results/task1_5_c_{}.mat'.format(k))
        os.rename('task1_5_idx_{}.mat'.format(k), '../results/task1_5_idx_{}.mat'.format(k))
        os.rename('task1_5_sse_{}.mat'.format(k), '../results/task1_5_sse_{}.mat'.format(k))

    verify_task1_5()

    return runtime


def run_task1_6(visual):

    Ks = Ks_clustering
    runtimes = []

    for k in Ks:
        fname = '../results/task1_5_c_{}.mat'.format(k)
        start_time = time()
        task1_6(fname)
        runtime = time() - start_time
        runtimes.append(runtime)
        plt.savefig(fname='task1_6_imgs_{}.pdf'.format(k))
        plt.show()

    return sum(runtimes)


def run_task1_7(visual):

    runtimes = []
    start_time = time()

    Ks = [1, 2, 3, 5, 10]
    MAT_M = '../results/task1_2_M.mat'
    MAT_evecs = '../results/task1_3_evecs.mat'
    MAT_evals = '../results/task1_3_evals.mat'
    posVec = my_mean(Xtrn)
    for k in Ks:
        MAT_ClusterCentres = '../results/task1_5_c_{}.mat'.format(k)
        task1_7(MAT_ClusterCentres, MAT_M, MAT_evecs, MAT_evals, posVec, 200)
        plt.savefig(fname='../results/task1_7_{}.pdf'.format(k))
        plt.savefig(fname='../results/task1_7_{}.png'.format(k))
        runtimes.append(time() - start_time)
        if visual:
            plt.show()

    return sum(runtimes)


def run_task2_1(visual):

    Ks = Ks_class
    task2_1(Xtrn, Ytrn, Xtst, Ytst, Ks)

    for k in Ks:
        os.rename('task2_1_cm{}.mat'.format(k), '../results/task2_1_cm{}.mat'.format(k))


def run_task2_2(visual):

    Ks = [1, 3]

    MAT_evecs = '../results/task1_3_evecs.mat'
    MAT_evals = '../results/task1_3_evals.mat'
    posVec = my_mean(Xtrn)
    nbins = 200

    # TODO can do 15000 on DICE
    N = 2000

    for k in Ks:
        Dmap = task2_2(Xtrn[:N], Ytrn[:N], k, MAT_evecs, MAT_evals, posVec, nbins)
        plt.savefig(fname='task2_2_imgs_{}.pdf'.format(k))
        plt.savefig(fname='task2_2_imgs_{}.png'.format(k))
        sio.savemat(file_name='task2_2_dmap_{}.mat'.format(k), mdict={'Dmap': Dmap})
        if visual:
            plt.show()

    for k in Ks:
        os.rename('task2_2_dmap_{}.mat'.format(k), '../results/task2_2_dmap_{}.mat'.format(k))
        os.rename('task2_2_imgs_{}.pdf'.format(k), '../results/task2_2_imgs_{}.pdf'.format(k))
        os.rename('task2_2_imgs_{}.png'.format(k), '../results/task2_2_imgs_{}.png'.format(k))


def run_task2_3(visual):

    task2_3(Xtrn, Ytrn)


def run_task2_4(visual):

    Corrs = task2_4(Xtrn, Ytrn)
    sio.savemat(file_name='../results/task2_4_corrs.mat', mdict={'Corrs': Corrs})


def run_task2_5(visual):

    start_time = time()
    task2_5(Xtrn, Ytrn, Xtst, Ytst, 0.01)
    print 'Elapsed time: {}'.format(time()-start_time)
    os.rename('task2_5_cm.mat', '../results/task2_5_cm.mat')
    os.rename('task2_5_m10.mat', '../results/task2_5_m10.mat')
    os.rename('task2_5_cov10.mat', '../results/task2_5_cov10.mat')


def run_task2_6(visual):

    MAT_evecs = '../results/task1_3_evecs.mat'
    MAT_evals = '../results/task1_3_evals.mat'
    start_time = time()
    task2_6(Xtrn, Ytrn, 0.01, MAT_evecs, MAT_evals, my_mean(Xtrn), 200)
    print 'Elapsed time: {}'.format(time() - start_time)
    plt.savefig('../results/task2_6_img.pdf')
    plt.savefig('../results/task2_6_img.png')
    if visual:
        plt.show()

def run_task2_7(visual):

    ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    for ratio in ratios:
        R = ratio * 100
        cm, acc = task2_7(Xtrn, Ytrn, Xtst, Ytst, 0.01, ratio)
        print 'Ratio = {}%'.format(R)
        print 'acc = {}\n'.format(acc)
        sio.savemat('../results/task2_7_cm_{}.mat'.format(R), mdict={'CM': cm})


def run_task2_8(visual):

    Ls = [2, 5, 10]
    for L in Ls:
        print '-------------------- L = {} --------------------'.format(L)
        task2_8(Xtrn, Ytrn, Xtst, Ytst, 0.01, L)
        print
        os.rename('task2_8_cm_{}.mat'.format(L), '../results/task2_8_cm_{}.mat'.format(L))
        os.rename('task2_8_g{}_m1.mat'.format(L), '../results/task2_8_g{}_m1.mat'.format(L))
        os.rename('task2_8_g{}_cov1.mat'.format(L), '../results/task2_8_g{}_cov1.mat'.format(L))




def main():
    try:
        # options:
        # -f: full task
        # -s: specific sub tasks
        # -v: visual (opens plots in new windows)
        # -c: cached (data from files)
        # -t: test
        opts, args = getopt.getopt(sys.argv[1:], 'fsvct')
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    visual = False
    cached = False
    test = False

    if len(opts) == 0:
        run_all()

    # print opts
    # print args

    for o, a in opts:
        if o == '-v':
            visual = True
        elif o == '-c':
            cached = True
        elif o == '-t':
            test = True
        elif o == '-f':
            globals()['run_task' + args[0]](visual)
        elif o == '-s':
            for task in args:
                if task == 'k':
                    if len(args) > 1:
                        k = int(args[1])
                    else:
                        k = 10
                    print 'Running k-means (k = {})\n'.format(k)
                    run_k_means(visual, k, test)
                    break
                elif task == 'dist':
                    run_vec_dist(visual)
                else:
                    task_n = task[0]
                    subtask_n = task[2:]
                    func_name = 'run_task' + task_n + '_' + subtask_n
                    print 'Running task ' + task + '\n'
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
    print 'To run k-means clustering with k=<X> just type: '
    print '\"python2.7 run_cw -s k <X>\"'


if __name__ == '__main__':

    # Values of k for k-means clustering
    Ks_clustering = [1, 2, 3, 4, 5, 7, 10, 15, 20]
    # Ks_clustering = [2, 3, 7] # for shorter testing

    # Values of k for k-nn classification
    Ks_class = [1, 3, 5, 10, 20]

    # Import data
    Xtrn, Ytrn, Xtst, Ytst = load_my_data_set('../data')

    # Normalise
    Xtrn = Xtrn / 255.0
    Xtst = Xtst / 255.0

    # Run
    main()
