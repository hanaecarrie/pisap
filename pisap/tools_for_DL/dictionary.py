""" Dictionary.
"""

# Sys import
import os
import datetime
import time
import itertools
import random
import numpy as np

# Third party import
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.utils import check_random_state, gen_batches
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from pisap.base.utils import timer


def generate_dico(flat_patches_subjects, nb_atoms=100, alpha=1, n_iter=1,
                  fit_algorithm='lars', transform_algorithm='lars',
                  batch_size=100, n_jobs=6, verbose=1):
    """Learn the dictionary from the real/imaginary part or the module/phase of
    images from the training set
    Parameters
    ----------
        flat_patches -- dictionary of lists of 1d array of flat patches (floats)
            a list per subject
        nb_atoms -- int, number of components of the dictionary (default=100)
        alpha -- float, regulation term (default=1)
        n_iter -- int, number of iterations (default=1)
        fit_algorithm -- ='lars', transform_algorithm='lars',
batch_size=100, n_jobs=6, verbose=1

    Returns
    -------
        dico -- object
    """
    dico = MiniBatchDictionaryLearning(
           n_components=nb_atoms, alpha=alpha, n_iter=n_iter,
           fit_algorithm=fit_algorithm, transform_algorithm=transform_algorithm,
           n_jobs=n_jobs, verbose=1)
    rng = check_random_state(0)
    if verbose == 2:
        print "Dictionary Learning starting"
    t_start = time.time()
    for patches_subject in flat_patches_subjects:
        patches = list(itertools.chain(*patches_subject))
        if verbose == 1:
            print("[info] number of patches of "
                  "the subject: {0}".format(len(patches)))
        rng.shuffle(patches)
        batches = gen_batches(len(patches), batch_size)
        for batch in batches:
            t0 = time.time()
            dico.partial_fit(patches[batch][:1])
            duration = time.time() - t0
            if verbose == 2:
                print("[info] batch time: {0}".format(duration))
    t_end = time.time()
    if verbose == 1:
        print('[info] dictionary learnt in {0}'.format(timer(t_start, t_end)))
    return dico
