from numpy import shape, zeros, ceil, sqrt
import matplotlib.pyplot as plt
from load_my_data_set import *


def montage(X, colormap='gray'):
    count, length = shape(X)
    m = n = int(sqrt(length))
    mm = int(ceil(sqrt(count)))
    nn = mm
    M = zeros((mm * m, nn * n))

    image_id = 0
    for j in range(mm):
        for k in range(nn):
            if image_id >= count:
                break
            sliceM, sliceN = j * m, k * n
            M[sliceM:sliceM + m, sliceN:sliceN + n] = np.reshape(X[image_id, :], (m, n))
            image_id += 1

    plt.imshow(M, cmap=colormap)
    plt.axis('off')
    # plt.show() TODO: I commented this out so I can save the figure
    # And I added this
    return plt.gcf(), plt.gca()

