def my_gaussian_classify(Xtrn, Ctrn, Xtst, epsilon):
	# Input:
	# Xtrn: M-by-D training data matrix
	# Ctrn: M-by-1 label vector for Xtrn
	# Xtst: N-by-D test data matrix
	# epsilon: A scalar parameter for regularisation
	# Output:
	# Cpreds: N-by-1 matrix of predicted labels for Xtst
	# Ms: D-by-K matrix of mean vectors
	# Covs: D-by-D-by-K 3D array of covariance matrices

	Cpreds = []
	Ms = []
	Covs = []
	pass # YourCode - Bayes classification with multivariate Gaussian distributions.#

	return Cpreds, Ms, Covs