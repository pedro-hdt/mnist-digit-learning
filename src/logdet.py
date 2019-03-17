import numpy as np


def logdet(A):
	# log(det(A)) where A is positive-definite.
	# This is faster and more stable than using log(det(A)).

	# From Tom Minka's lightspeed toolbox

	U = np.linalg.cholesky(A)
	y = 2*np.sum(np.log(np.diag(U)))

	return y
