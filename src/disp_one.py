import matplotlib.pyplot as plt
import numpy as np


def disp_one(X, Y):
	for i in range(len(X)):
		print(Y[i])
		plt.imshow(np.reshape(X[i, :], (28, 28)), cmap='gray')
		plt.show()
		raw_input('Hit return: ')

	print('Done')
