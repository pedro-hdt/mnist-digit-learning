import numpy as np


def rd(fname):
    fid = open(fname, 'r')
    header = np.fromfile(fid, count=4, dtype='>i')
    n_images = header[1]
    height = header[2]
    width = header[3]
    bsize = height * width

    images = np.fromfile(fid, dtype='uint8')
    images = np.reshape(images, (n_images, bsize))
    fid.close()
    return images