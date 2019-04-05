from load_my_data_set import load_my_data_set
from montage import montage
import task2
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from aux import my_mean
from time import time
import os

# Values of k for k-nn classification
Ks_class = [1, 3, 5, 10, 20]

# Import data
Xtrn, Ytrn, Xtst, Ytst = load_my_data_set('../data')

# Normalise
Xtrn = Xtrn / 255.0
Xtst = Xtst / 255.0

overall_start_time = time()

# 2.1 ===================================================================================
Ks = Ks_class
task2.task2_1(Xtrn, Ytrn, Xtst, Ytst, Ks)
for k in Ks:
    os.rename('task2_1_cm{}.mat'.format(k), '../results/task2_1_cm{}.mat'.format(k))

# 2.2 ===================================================================================
Ks = [1, 3]

MAT_evecs = '../results/task1_3_evecs.mat'
MAT_evals = '../results/task1_3_evals.mat'
posVec = np.atleast_2d(my_mean(Xtrn))
nbins = 200

# This can run 15000 on DICE but takes a while
# To avoid errors we stick to the suggested subset of 2000
N = 2000

for k in Ks:
    Dmap = task2.task2_2(Xtrn[:N], Ytrn[:N], k, MAT_evecs, MAT_evals, posVec, nbins)
    plt.savefig(fname='task2_2_imgs_{}.pdf'.format(k))
    plt.savefig(fname='task2_2_imgs_{}.png'.format(k))
    sio.savemat(file_name='task2_2_dmap_{}.mat'.format(k), mdict={'Dmap': Dmap})

for k in Ks:
    os.rename('task2_2_dmap_{}.mat'.format(k), '../results/task2_2_dmap_{}.mat'.format(k))
    os.rename('task2_2_imgs_{}.pdf'.format(k), '../results/task2_2_imgs_{}.pdf'.format(k))
    os.rename('task2_2_imgs_{}.png'.format(k), '../results/task2_2_imgs_{}.png'.format(k))

# 2.3 ===================================================================================
task2.task2_3(Xtrn, Ytrn)
plt.savefig('../results/task2_3_img.pdf')
plt.savefig('../results/task2_3_img.png')

# 2.4 ===================================================================================
Corrs = task2.task2_4(Xtrn, Ytrn)
sio.savemat(file_name='../results/task2_4_corrs.mat', mdict={'Corrs': Corrs})
print Corrs

# 2.5 ===================================================================================
start_time = time()
task2.task2_5(Xtrn, Ytrn, Xtst, Ytst, 0.01)
print 'Elapsed time (2.5): {}'.format(time() - start_time)
os.rename('task2_5_cm.mat', '../results/task2_5_cm.mat')
os.rename('task2_5_m10.mat', '../results/task2_5_m10.mat')
os.rename('task2_5_cov10.mat', '../results/task2_5_cov10.mat')

# 2.6 ===================================================================================
MAT_evecs = '../results/task1_3_evecs.mat'
MAT_evals = '../results/task1_3_evals.mat'
start_time = time()
posVec = np.atleast_2d(my_mean(Xtrn))
Dmap = task2.task2_6(Xtrn, Ytrn, 0.01, MAT_evecs, MAT_evals, posVec, 200)
sio.savemat('../results/task2_6_dmap.mat', mdict={'Dmap': Dmap})
print 'Elapsed time (2.6): {}'.format(time() - start_time)
plt.savefig('../results/task2_6_img.pdf')
plt.savefig('../results/task2_6_img.png')

# 2.7 ===================================================================================
ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
for ratio in ratios:
    R = int(ratio * 100)
    cm, acc = task2.task2_7(Xtrn, Ytrn, Xtst, Ytst, 0.01, ratio)
    print 'Ratio = {}%'.format(R)
    print 'acc = {}\n'.format(acc)
    sio.savemat('../results/task2_7_cm_{}.mat'.format(R), mdict={'CM': cm})

# 2.8 ===================================================================================
Ls = [2, 5, 10]
for L in Ls:
    print '-------------------- L = {} --------------------'.format(L)
    task2.task2_8(Xtrn, Ytrn, Xtst, Ytst, 0.01, L)
    print
    os.rename('task2_8_cm_{}.mat'.format(L), '../results/task2_8_cm_{}.mat'.format(L))
    os.rename('task2_8_g{}_m1.mat'.format(L), '../results/task2_8_g{}_m1.mat'.format(L))
    os.rename('task2_8_g{}_cov1.mat'.format(L), '../results/task2_8_g{}_cov1.mat'.format(L))

# FINAL TIMING ==========================================================================
runtime = time() - overall_start_time
print '\nTotal runtime of task 2: ', runtime

# FINAL PLOTIING ========================================================================
plt.show()