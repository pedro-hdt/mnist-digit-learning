from load_my_data_set import *
from task1_1 import *


# Import data
Xtrn, Ytrn, Xtst, Ytst = load_my_data_set('../data')


# Normalise
Xtrn = Xtrn / 255.0
Xtst = Xtst / 255.0

task1_1(Xtrn, Ytrn)
