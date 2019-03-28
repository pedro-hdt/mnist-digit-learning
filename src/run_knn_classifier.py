from my_dist import *


def run_knn_classifier(Xtrn, Ytrn, Xtst, Ks):
	"""
	Write a Python function for the classification with k-NN
	:param Xtrn: M-by-D training data matrix
	:param Ytrn: M-by-1 label vector for Xtrn
	:param Xtst: N-by-D test data matrix
	:param Ks: L-by-1 vector of the numbers of nearest neighbours in Xtrn
	:return: Ypreds : N-by-L matrix of predicted labels for Xtst
	"""

	Ypreds = []

	dist_mat = vec_sq_dist(Xtrn, Xtst)

	for k in Ks:

		k_nn = k

		for i in range(len(Xtst)):

			# curent test sample being classified
			test_sample = Xtst[i]

			# distance vector between test sample and each of training samples
			dist = dist_mat[:, i]





	return Ypreds