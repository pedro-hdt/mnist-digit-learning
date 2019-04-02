def my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon):
	"""
	Write a Python function for the classification with a single Gaussian distribution per class
	where Xtrn, Ytrn, Xtst, and Ypreds are the same as those in Task1. epsilon is a scalar
	(double) for the regularisation of covariance matrix described in Lecture 8, in which we add
	a small positive number (epsilon) to the diagonal elements of covariance matrix,
	i.e. SIGMA <-- SIGMA + (epsilon)I, where I is the identity matrix.

	Note that, as opposed to Tasks 2.3 and 2.4 that use PCA, we do not apply PCA to the
	data in this task.

	:param Xtrn: M-by-D training data matrix
	:param Ctrn: M-by-1 label vector for Xtrn
	:param Xtst: N-by-D test data matrix
	:param epsilon:
	:return:
		Cpreds: N-by-1 matrix of predicted labels for Xtst
		Ms: D-by-K matrix of mean vectors where Ms[k, :] is the sample
		mean vector for class k.
		Covs: D-by-D-by-K 3D array of covariance matrices, where Cov[k, :, :]
		is the covariance matrix (after the regularisation) for class k.
	"""

	Cpreds = []
	Ms = []
	Covs = []
	pass # YourCode - Bayes classification with multivariate Gaussian distributions.#

	return Cpreds, Ms, Covs