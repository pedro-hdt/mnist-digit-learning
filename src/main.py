from load_my_data_set import load_my_data_set
from montage import montage
import task1
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from aux import my_mean
from time import time
import os

# Values of k for k-means clustering
Ks_clustering = [1, 2, 3, 4, 5, 7, 10, 15, 20]

# Values of k for k-nn classification
Ks_class = [1, 3, 5, 10, 20]

# Import data
Xtrn, Ytrn, Xtst, Ytst = load_my_data_set('../data')

# Normalise
Xtrn = Xtrn / 255.0
Xtst = Xtst / 255.0

# 1.1 ===================================================================================
plt.clf()
task1.task1_1(Xtrn, Ytrn)

# 1.2 ===================================================================================
plt.clf()
start_time = time()
M = task1.task1_2(Xtrn, Ytrn)
runtime = time() - start_time
plt.savefig(fname='../results/task1_2_imgs.pdf')
plt.savefig(fname='../results/task1_2_imgs.png')
sio.savemat(file_name='../results/task1_2_M.mat', mdict={'M': M})

# 1.3 ===================================================================================
start_time = time()
EVecs, EVals, CumVar, MinDims = task1.task1_3(Xtrn)
runtime = time() - start_time
sio.savemat(file_name='../results/task1_3_evecs.mat', mdict={'EVecs': EVecs})
sio.savemat(file_name='../results/task1_3_evals.mat', mdict={'EVals': EVals})
sio.savemat(file_name='../results/task1_3_cumvar.mat', mdict={'CumVar': CumVar})
sio.savemat(file_name='../results/task1_3_mindims.mat', mdict={'MinDims': MinDims})

# 1.4 ===================================================================================
EVecs, EVals, CumVar, MinDims = task1.task1_3(Xtrn)
plt.clf()
start_time = time()
task1.task1_4(EVecs)
runtime = time() - start_time
fig = plt.gcf()
fig.suptitle('First 10 Principal Components')
fig.canvas.set_window_title('Task 1.4')
plt.savefig(fname='../results/task1_4_imgs.pdf')
plt.savefig(fname='../results/task1_4_imgs.png')

# 1.5 ===================================================================================
Ks = Ks_clustering

# Call the function and time it
start_time = time()
task1.task1_5(Xtrn, Ks)
runtime = time() - start_time
print 'Elapsed time: {} secs'.format(runtime)

# Move files to results directory to avoid cluttering our src folder
print 'Moving files...'
for k in Ks:
    os.rename('task1_5_c_{}.mat'.format(k), '../results/task1_5_c_{}.mat'.format(k))
    os.rename('task1_5_idx_{}.mat'.format(k), '../results/task1_5_idx_{}.mat'.format(k))
    os.rename('task1_5_sse_{}.mat'.format(k), '../results/task1_5_sse_{}.mat'.format(k))

# 1.6 ===================================================================================
Ks = Ks_clustering
runtimes = []

for k in Ks:
    fname = '../results/task1_5_c_{}.mat'.format(k)
    task1.task1_6(fname)
    fig = plt.gcf()
    fig.suptitle('Cluster centres for k = {}'.format(k))
    fig.canvas.set_window_title('Task 1.6')
    plt.savefig(fname='../results/task1_6_imgs_{}.pdf'.format(k))
    plt.savefig(fname='../results/task1_6_imgs_{}.png'.format(k))
    plt.figure()
    montage(sio.loadmat('../results/task1_5_libc_{}.mat'.format(k))['C'])

# 1.7 ===================================================================================
Ks = [1, 2, 3, 5, 10]
MAT_M = '../results/task1_2_M.mat'
MAT_evecs = '../results/task1_3_evecs.mat'
MAT_evals = '../results/task1_3_evals.mat'
posVec = np.atleast_2d(my_mean(Xtrn))
for k in Ks:
    MAT_ClusterCentres = '../results/task1_5_c_{}.mat'.format(k)
    task1.task1_7(MAT_ClusterCentres, MAT_M, MAT_evecs, MAT_evals, posVec, 200)
    plt.savefig(fname='../results/task1_7_{}.pdf'.format(k))
    plt.savefig(fname='../results/task1_7_{}.png'.format(k))
    runtimes.append(time() - start_time)

