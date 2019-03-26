from montage import *
from load_my_data_set import *
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


def test_task1_1(visual):
    plt.clf()
    task1_1(Xtrn, Ytrn)
    if visual:
        plt.show()


def test_task1_2(visual):
    plt.clf()
    M = task1_2(Xtrn, Ytrn)
    plt.savefig(fname='../results/task1_2_imgs.pdf')
    sio.savemat(file_name='../results/task1_2_M.mat', mdict={'M': M})
    if visual:
        plt.show()


def test_task1_3(visual):
    plt.clf()
    EVecs, EVals, CumVar, MinDims = task1_3(Xtrn)
    if visual:
        plt.show()
    sio.savemat(file_name='../results/task1_3_evecs.mat', mdict={'EVecs': EVecs})
    sio.savemat(file_name='../results/task1_3_evals.mat', mdict={'EVals': EVals})
    sio.savemat(file_name='../results/task1_3_cumvar.mat', mdict={'CumVar': CumVar})
    sio.savemat(file_name='../results/task1_3_mindims.mat', mdict={'MinDims': MinDims})


def test_task1_4(visual):
    plt.clf()
    EVecs, EVals, CumVar, MinDims = task1_3(Xtrn)
    plt.clf()
    task1_4(EVecs)
    plt.suptitle('First 10 Principal Components')
    plt.savefig(fname='../results/task1_4_imgs.pdf')
    if visual:
        plt.show()


def test_task1_5():
    Ks = [1, 2, 3, 4, 5, 7, 10, 15, 20]
    task1_5(Xtrn, Ks)


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'fsv')
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    if len(args) > 1:
        print 'Too many arguments.'
        usage()
        sys.exit(2)

    visual = False

    for o, a in opts:
        if o == '-v':
            visual = True
        elif o == '-f':
            test_task1_1(visual)
            test_task1_2(visual)
        elif o == '-s':
            if len(args) != 1:
                usage()
                assert False, 'Invalid number of arguments!'
            subtask = args[0]
            globals()['test_task1_' + subtask](visual)
        else:
            assert False, "Invalid option!"


def usage():
    print 'Usage: \"python2.7 test_task1\" followed by: '
    print '-v to open the plots as the task runs'
    print '-f for the full task'
    print '-s for a specific sub-task followed by the sub-task index'
    print 'e.g. for task 1.1: '
    print '\"python2.7 test_task1 -s 1\"'


if __name__ == '__main__':

    # Import data
    Xtrn, Ytrn, Xtst, Ytst = load_my_data_set('../data')

    # Normalise
    Xtrn = Xtrn / 255.0
    Xtst = Xtst / 255.0

    # Run
    main()
