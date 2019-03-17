#
# A template to load the data files
#
#
import numpy as np


def rd_mnist_labels(fname):
	fid = open(fname, 'r')
	header = np.fromfile(fid, count=2, dtype='>i')
	labels = np.fromfile(fid, dtype='uint8')
	fid.close()
	return labels, header


def rd_mnist_images(fname):
	fid = open(fname, 'r')
	header = np.fromfile(fid, count=4, dtype='>i')
	n_images = header[1]
	height = header[2]
	width = header[3]
	bsize = height * width
	images = np.fromfile(fid, dtype='uint8')
	images = np.reshape(images, (n_images, bsize))
	fid.close()
	return images, header


def load_my_data_set(dir):
	Xtrn, _ = rd_mnist_images(dir+'/trn-images-idx3-ubyte')
	Ytrn, _ = rd_mnist_labels(dir+'/trn-labels-idx1-ubyte')
	Xtst, _ = rd_mnist_images(dir+'/tst-images-idx3-ubyte')
	Ytst, _ = rd_mnist_labels(dir+'/tst-labels-idx1-ubyte')

	return Xtrn, Ytrn, Xtst, Ytst