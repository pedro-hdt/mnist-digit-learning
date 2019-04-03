from my_mean import *
from logdet import *


def my_gaussian_classify(Xtrn, Ytrn, Xtst, epsilon):
	"""
	Write a Python function for the classification with a single Gaussian distribution per class
	where Xtrn, Ytrn, Xtst, and Ypreds are the same as those in Task1. epsilon is a scalar
	(double) for the regularisation of covariance matrix described in Lecture 8, in which we add
	a small positive number (epsilon) to the diagonal elements of covariance matrix,
	i.e. SIGMA <-- SIGMA + (epsilon)I, where I is the identity matrix.

	Note that, as opposed to Tasks 2.3 and 2.4 that use PCA, we do not apply PCA to the
	data in this task.

	:param Xtrn: M-by-D training data matrix
	:param Ytrn: M-by-1 label vector for Xtrn
	:param Xtst: N-by-D test data matrix
	:param epsilon:
	:return:
		Cpreds: N-by-1 matrix of predicted labels for Xtst
		Ms: K-by-D matrix of mean vectors where Ms[k, :] is the sample
		mean vector for class k.
		Covs: K-by-D-by-D 3D array of covariance matrices, where Cov[k, :, :]
		is the covariance matrix (after the regularisation) for class k.
	"""

	# Number of classes
	K = 10

	# Size of the matrices
	N = len(Xtst)
	M = len(Xtrn)
	D = len(Xtrn[0])

	Cpreds = np.zeros((N, 1))
	Ms = np.zeros((K, D))
	Covs = np.zeros((K, D, D))
	inv_Covs = np.zeros((K, D, D))
	priors = np.zeros(K)

	# Bayes classification with multivariate Gaussian distributions
	for C_k in range(K):

		# Extract class samples
		class_samples = Xtrn[Ytrn[:] == C_k]

		# Calculate prior probabilities
		priors[C_k] = float(len(class_samples)) / M

		# Estimate mean
		Ms[C_k] = my_mean(class_samples)

		# Estimate covariance matrix
		X_shift = class_samples - Ms[C_k]
		Covs[C_k] = (1.0 / len(class_samples)) * np.dot(X_shift.T, X_shift) + epsilon * np.identity(D)
		inv_Covs[C_k] = np.linalg.inv(Covs[C_k])

	# For each class, calculate the log posterior probability for all samples
	log_post_probs = np.zeros((K, N))
	for C_k in range(K):

		# Extract the mean and mean shift the data
		mu = Ms[C_k]
		Xtst_shift = Xtst - mu

		# Formula derived from (with extra vectorization) lect note 9 (equation 9.9 on page 4)
		# https://www.inf.ed.ac.uk/teaching/courses/inf2b/learnnotes/inf2b-learn09-notes-nup.pdf
		log_post_probs[C_k] = \
			- 0.5 * np.diag(np.dot(np.dot(Xtst_shift, inv_Covs[C_k]), Xtst_shift.T)) \
			- 0.5 * logdet(Covs[C_k]) \
			+ np.log(priors[C_k])

	# Finally, assign to each sample the class that maximises the log posterior probaility
	Cpreds = log_post_probs.argmax(axis=0)

	return Cpreds, Ms, Covs