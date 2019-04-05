import scipy.io as sio
from montage import *


def task1_6(MAT_ClusterCentres):
    """
    Write a Pyhton function that displays the image of each cluster centre, where you should use
    montage() function to put all the images into a single figure.

    Input:
    :param MAT_ClusterCentres: file name of the file that contains cluster centres C.
    """

    C = sio.loadmat(file_name=MAT_ClusterCentres)['C']
    montage(C)
