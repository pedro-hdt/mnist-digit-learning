from montage import *
from load_my_data_set import *
from task1_1 import *
from task1_2 import *
import sys, getopt


def test_task1_1(visual=False): #TODO: implement -v flag for visual (with windows opening)
    plt.clf()
    task1_1(Xtrn, Ytrn)
    if visual:
        plt.show()


def test_task1_2():
    plt.clf()
    M = task1_2(Xtrn, Ytrn)
    plt.savefig(fname='../results/task1_2_imgs.pdf')
    plt.show()
    sio.savemat(file_name='../results/task1_2_M.mat', mdict={'M': M})

def main():

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'fs')
    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    if len(args) > 1:
        print 'Too many arguments.'
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == '-f':
            test_task1_1()
            test_task1_2()
        elif o == '-s':
            if len(args) != 1:
                usage()
                assert False, 'Invalid number of arguments!'
            subtask = args[0]
            globals()['test_task1_' + subtask]()
        else:
            assert False, "Invalid option!"


def usage():
    print 'Usage: \"python2.7 test_task1\" followed by: '
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
