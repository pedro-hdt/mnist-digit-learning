from my_dist import *
from scipy.stats import mode

def run_knn_classifier(Xtrn, Ytrn, Xtst, Ks):
	"""
	Write a Python function for the classification with k-NN
	:param Xtrn: M-by-D training data matrix
	:param Ytrn: M-by-1 label vector for Xtrn
	:param Xtst: N-by-D test data matrix
	:param Ks: L-by-1 vector of the numbers of nearest neighbours in Xtrn
	:return: Ypreds : N-by-L matrix of predicted labels for Xtst
	"""

	N = len(Xtst)
	L = len(Ks)
	Ypreds = np.zeros((N, L), dtype='uint8')

	dist_mat = vec_sq_dist(Xtrn, Xtst)

	for l in range(L):
		k = Ks[l] # k in k-NN
		for n in range(N):
			# curent test sample being classified
			tst_sample = Xtst[n]

			# distance vector between test sample and each of training samples
			dist = dist_mat[:, n]

			# get sorting index for distances to first k-nn only
			idx = dist.argsort()[:k]

			# extract list of closest training samples
			k_nn_labels = Ytrn[idx]

			# we predict it is the most common value (ie the mode) of the k-nn
			Ypreds[n, l], _ = mode(k_nn_labels)

	return Ypreds