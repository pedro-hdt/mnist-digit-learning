
--- How to show the data set ---
Write the follow python script to load the data
    from load_my_data_set import *
    from disp_one import *

    # replace <UUN> with your UUN in the following.
    dset_dir = '/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/<UUN>'
    Xtrn, Ytrn, Xtst, Ytst = load_my_data_set(dset_dir)
    disp_one(Xtrn, Ytrn)