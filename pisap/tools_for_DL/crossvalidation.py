# coding: utf8

""" Crossvalidation.
"""

# Sys import
import os
import datetime
import time
import itertools
import random
import numpy as np
import matplotlib.pyplot as plt

# Third party import
import scipy.io
import scipy.fftpack as pfft
from scipy.misc import imsave
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.utils import check_random_state, gen_batches
import pickle

# Package import
from pisap.base.utils import min_max_normalize
from pisap.base.utils import extract_patches_from_2d_images
from pisap.numerics.linear import DictionaryLearningWavelet
from pisap.numerics.fourier import NFFT, FFT
from pisap.numerics.gridsearch import grid_search
from pisap.numerics.cost import _preprocess_input



def save_object(obj, filename):
    """Save object into pickle format
    ---------
    Inputs:
    obj -- variable, name of the object in the current workspace
    filename -- string, filename to save the object
    ---------
    Outputs:
    nothing
    """
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    """Load object saved into pickle format
    ---------
    Inputs:
    filename -- string, path where the pickle object is saved
    ---------
    Outputs:
    nothing
    """
    with open(filename, 'rb') as pfile:
        return pickle.load(pfile)


def generate_dico(flat_patches_subjects, nb_atoms, alpha, n_iter,
                  fit_algorithm='lars', transform_algorithm='lars',
                  batch_size=100, n_jobs=-1, verbose=False):
    """Learn the dictionary from the real/imaginary part or the module/phase of
    images from the training set
    Parameters
    ----------
        flat_patches -- dictionary of lists of 1d array of flat patches (floats)
            a list per subject
        nb_atoms -- int, number of components of the dictionary
        alpha -- float, regulation term (default=1)
        n_iter -- int, number of iterations (default=100)

    Returns
    -------
        dico -- object
    """
    dico = MiniBatchDictionaryLearning(
           n_components=nb_atoms, alpha=alpha, n_iter=n_iter,
           fit_algorithm=fit_algorithm, transform_algorithm=transform_algorithm,
           n_jobs=n_jobs, verbose=1)
    rng = check_random_state(0)
    if verbose:
        print "Dictionary Learning starting"
    t_start = time.time()
    for patches_subject in flat_patches_subjects:
        patches = list(itertools.chain(*patches_subject))
        if verbose:
            print("[info] number of patches of "
                  "the subject: {0}".format(len(patches)))
        rng.shuffle(patches)
        batches = gen_batches(len(patches), batch_size)
        for batch in batches:
            t0 = time.time()
            dico.partial_fit(patches[batch][:1])
            duration = time.time() - t0
            if verbose:
                print("[info] batch time: {0}".format(duration))
    t_end = time.time()
    if verbose:
        print('[info] dictionary learnt in {0}'.format(timer(t_start, t_end)))
    return dico


def generate_dicos_forCV(nb_cv_iter, saving_path, nb_subjects,
                         imgs, img_shape, param_grid, n_iter_dico, batch_size,
                         fit_algorithm='lars', transform_algorithm='lars',
                         verbose=False):
    """ Learn dicos from the IMGS over a grid of parameters
        Create folders and files to save the learnt dicos.

    Parameters
    ----------
    nb_cv_iter: int, number of iterations of the crossvalidation.
    saving_path: str, path to save the results
    size_training_set_per_subject: int, number of images per suy
    size_validation_set: int
    param_grid: dictionary{'nb_atoms':[200,400,600],
                      'patch_size':[7,10],
                      'alpha':[1e-2],
                      }
    n_iter_dico: int
        'batch_size':5000,
        'fit_algorithm':'lars',
        'transform_algorithm':'lars',
        'imgs': imgs_constrast_5,
        'img_shape':imgs[0].shape
    """
    # create folder
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    crossval_dir = saving_path + date
    os.makedirs(crossval_dir)
    list_args = [dict(zip(param_grid, x))
                   for x in itertools.product(*param_grid.values())]

    for i in range(1, nb_cv_iter + 1):
        # create subfolder iter, subsubfolder refs_train and dicos
        title_iter = os.path.join(crossval_dir, 'iter_'+str(i))
        os.makedirs(title_iter)
        title_dico = os.path.join(title_iter, 'dicos')
        os.makedirs(title_dico)

        # provide training set
        num_subject = random.randint(1, nb_subjects)
        if verbose:
            print("[info] crossval iteration number: {0}".format(num_subject))
        os.makedirs(os.path.join(title_iter,
                                 'validation_subject_'+str(num_subject)))
        save_object(num_subject, os.path.join(title_iter,
                                 'validation_subject_'+str(num_subject),
                                 'num_validation_subject'))
        training_set = imgs[:]
        if verbose:
            print("[info] index validation subject: {0}".format(num_subject))
        training_set.pop(num_subject-1)
        for args in list_args:
            if verbose:
                print("[info] dictionary parameters: {0}".format(args))
            for key, value in args.iteritems():
                exec(key + '=value')
            dico_folder = os.path.join(title_dico, 'dico')
            for key, value in args.iteritems():
                dico_folder += "_{}={}".format(key, value)
            os.makedirs(dico_folder)
            save_object(args, os.path.join(dico_folder, 'dico_parameters.pkl'))

            # preprocessing data, learning and saving dictionaries
            flat_patches_real = generate_flat_patches(training_set, patch_size,
                                                      'real')
            dico_real = generate_dico(flat_patches_real, nb_atoms, alpha,
                                      n_iter_dico,
                                      fit_algorithm=fit_algorithm,
                                      transform_algorithm=transform_algorithm,
                                      batch_size=batch_size)
            save_object(dico_real, dico_folder+'/dico_real.pkl')
            flat_patches_imag = generate_flat_patches(training_set, patch_size,
                                                      'imag')
            dico_imag = generate_dico(flat_patches_imag, nb_atoms, alpha,
                                      n_iter_dico,
                                      fit_algorithm=fit_algorithm,
                                      transform_algorithm=transform_algorithm,
                                      batch_size=batch_size)
            save_object(dico_imag, dico_folder+'/dico_imag.pkl')
    if verbose:
        print('[info] generate_dicos_forCV successfully ended!')


def compute_sparse_code_forCV(CV_path, threshold=0, verbose=False):
    """ Load the previousely learnt dictionaries
        Compute the sparse code for the validation_set
        Reconstruct the image from the sparse code

    Parameters:
    -----
        CV_path: str, path to the crossvalidation folder
        threshold (default =0): thresholding level of the sparse coefficents
    Return:
    -------
        str, massage to indicate that the algorithm successfully ended
    """
    nb_cv_iter = len(os.listdir(CV_path))

    for i in range(1, nb_cv_iter + 1):
        dicos_list = os.listdir(os.path.join(CV_path, 'iter_'+str(i), 'dicos'))
        num_validation_subject = load_object(os.path.join(CV_path,
                                 'iter_'+str(i),
                                 'validation_subject_'+str(num_subject),
                                 'num_validation_subject'))
        validation_set = validation_set.tolist()
        for val in validation_set:
            val = np.array(val)
        img_shape = np.array(validation_set[0]).shape
        index_val = np.load(os.path.join(CV_path, 'iter_'+str(i), 'refs_val',
                                         'index_val.npy'))

        for dico in dicos_list:
            dico_folder = os.path.join(CV_path, 'iter_'+str(i), 'dicos', dico)
            dico_real = load_object(os.path.join(dico_folder, 'dico_real.pkl'))
            dico_imag = load_object(os.path.join(dico_folder, 'dico_imag.pkl'))
            list_args = load_object(os.path.join(dico_folder,
                                                'dico_parameters.pkl'))
            patch_size = list_args['patch_size']
            flat_patches_real = generate_flat_patches(validation_set,
                                                      patch_size, 'real')
            flat_patches_imag = generate_flat_patches(validation_set,
                                                      patch_size, 'imag')
            recons_real = reconstruct_2d_images_from_flat_patched_images(
                flat_patches_real, dico_real, img_shape, threshold)
            recons_imag = reconstruct_2d_images_from_flat_patched_images(
                flat_patches_imag, dico_imag, img_shape, threshold)

            for j in range(len(validation_set)):
                recons = recons_real[j]+1j*recons_imag[j]
                #saving_reference
                path_reconstruction = os.path.join(dico_folder,
                    'reconstructions', 'ind_'+str(index_val[j]))
                os.makedirs(path_reconstruction)
                np.save(os.path.join(path_reconstruction, 'ref.npy'),
                        validation_set[j])
                scipy.io.savemat(os.path.join(path_reconstruction, 'ref.mat'),
                                 mdict={'ref': validation_set[j]})
                imsave(os.path.join(path_reconstruction, 'ref.png'),
                       min_max_normalize(np.abs(validation_set[j])))
                #reconstructed image
                scipy.io.savemat(os.path.join(path_reconstruction,
                                 'sparse_code_recons.mat',
                                 mdict={'sparse_code_recons': recons}))
                imsave(os.path.join(path_reconstruction,
                                    'sparse_code_recons.png'),
                                    min_max_normalize((np.abs(recons))))
    if verbose:
        print('[info] compute_sparse_code_forCV successfully ended!')


def compute_pisap_gridsearch_forCV(CV_path, sampling_scheme, mu_grid, func,
                                   pisap_parameters_grid, Fourier_option,
                                   ft_obj, verbose=False):
    """ Load the previousely learnt dictionaries
        Compute the sparse code for the validation_set
        Reconstruct the image from the sparse code

    Parameters:
    -----
        CV_path: str, path to the crossvalidation folder
        params: dictionary of parameters
            example:     {'sampling_scheme': 2d np.array if FFT,
                          'Fourier_option': FFT,
                          'ft_obj': FFT(samples_locations=samples,
                                        img_shape = img_shape),
                          'func':sparse_rec_condat_vu,
                          'pisap_parameters_grid': pisap_parameters like
                                                   for grid_search,
                         }
    Return:
    -------
        str, massage to indicate that the algorithm successfully ended
    """
    #params to variables
    nb_cv_iter = len(os.listdir(os.path.join(CV_path)))
    is_first_iter = True

    for i in range(1, nb_cv_iter + 1):
        if verbose:
            print("[info] crossval iteration: {0}".format(i))
        dicos_list = os.listdir(os.path.join(CV_path, 'iter_'+str(i), 'dicos'))
        validation_set = np.load(os.path.join(CV_path, 'iter_'+str(i),
                                              'refs_val', 'refs_val.npy'))
        validation_set = validation_set.tolist()
        kspaceCS = []
        for val in validation_set:
            val = np.array(val)
            if Fourier_option == FFT:
                kspaceCS_i = np.fft.fft2(val)*pfft.fftshift(sampling_scheme)
            elif Fourier_option == NFFT:
                kspaceCS_i = np.fft.fft2(val)*(sampling_scheme)
            kspaceCS_i = pfft.fftshift(kspaceCS_i)
            kspaceCS.append(kspaceCS_i)
        img_shape = np.array(validation_set[0]).shape
        index_val = np.load(os.path.join(CV_path, 'iter_'+str(i),
                                         'refs_val/index_val.npy'))
        for dico in dicos_list:
            dico_folder = os.path.join(CV_path, 'iter_'+str(i), 'dicos', dico)
            dico_real = load_object(os.path.join(dico_folder, 'dico_real.pkl'))
            dico_imag = load_object(os.path.join(dico_folder, 'dico_imag.pkl'))
            list_args = load_object(os.path.join(dico_folder,
                                                'dico_parameters.pkl'))
            patch_size = list_args['patch_size']
            DLW_r = DictionaryLearningWavelet(dico_real, dico_real.components_,
                                              img_shape)
            DLW_i = DictionaryLearningWavelet(dico_imag, dico_imag.components_,
                                              img_shape)
            if is_first_iter: #TODO debug this
                pisap_parameters_grid['linear_kwargs']['DLW_r'] = DLW_r
                pisap_parameters_grid['linear_kwargs']['DLW_i'] = DLW_i
            else:
                pisap_parameters_grid['linear_kwargs'][0]['DLW_r'] = DLW_r
                pisap_parameters_grid['linear_kwargs'][0]['DLW_i'] = DLW_i

            for j in range(len(validation_set)):
                if verbose:
                    print("[info] index of current validation image:"
                          "{0}".format(j))
                path_reconstruction = os.path.join(dico_folder,
                                                   'reconstructions',
                                                   'ind_'+str(index_val[j]))
                os.makedirs(path_reconstruction)
                ref = np.abs(validation_set[j])
                scipy.io.savemat(path_reconstruction+'/ref.mat',
                                 mdict={'ref': validation_set[j]})
                imsave(os.path.join(path_reconstruction, 'ref.png'),
                    min_max_normalize(np.abs(validation_set[j])))
                if is_first_iter: #TODO debug this
                    pisap_parameters_grid['metrics']['ssim']['cst_kwargs']['ref'] = ref
                    pisap_parameters_grid['metrics']['snr']['cst_kwargs']['ref'] = ref
                    pisap_parameters_grid['metrics']['psnr']['cst_kwargs']['ref'] = ref
                    pisap_parameters_grid['metrics']['nrmse']['cst_kwargs']['ref'] = ref
                else:
                    pisap_parameters_grid['metrics'][0]['ssim']['cst_kwargs']['ref'] = ref
                    pisap_parameters_grid['metrics'][0]['snr']['cst_kwargs']['ref'] = ref
                    pisap_parameters_grid['metrics'][0]['psnr']['cst_kwargs']['ref'] = ref
                    pisap_parameters_grid['metrics'][0]['nrmse']['cst_kwargs']['ref'] = ref

                is_first_iter = False
                data_undersampled = ft_obj.op((validation_set[j]))
                pisap_parameters_grid['data'] = data_undersampled
                # zero-order solution
                scipy.io.savemat(os.path.join(path_reconstruction,
                    'zero_order_solution.mat'), mdict={'zero_order_solution':
                    ft_obj.adj_op(data_undersampled)})
                imsave(os.path.join(
                    path_reconstruction, 'zero_order_solution.png'),
                    min_max_normalize(np.abs(ft_obj.adj_op(data_undersampled))))
                # computing gridsearch
                list_kwargs, res = grid_search(func, pisap_parameters_grid,
                                               do_not_touch=[],
                                               n_jobs=-2,
                                               verbose=1)
                save_object(list_kwargs, os.path.join(path_reconstruction,
                                                      'list_kwargs.pkl'))
                save_object(res, os.path.join(path_reconstruction, 'res.pkl'))
    if verbose:
        print('[info] compute_pisap_gridsearch_forCV successfully ended!')

def compute_mask_forCV(CV_path, verbose=False):
    """ Compute mask
        Compute the sparse code for the validation_set
        Reconstruct the image from the sparse code

    Parameters
    ----------
        CV_path: str, path to the crossvalidation folder

    Return
    -------
        str, massage to indicate that the algorithm successfully ended
    """
    nb_cv_iter = len(os.listdir(CV_path))
    for i in range(1, nb_cv_iter + 1):
        if verbose:
            print("[info] crossval iter number {0}".format(i))
        path = os.path.join(CV_path, 'iter_'+str(i), 'dicos')
        dicos = os.listdir(path)
        for k in range(len(dicos)):
            dico_name = dicos[k]
            path_dico = os.path.join(CV_path, 'iter_'+str(i), 'dicos',
                                     dico_name, 'reconstructions')
            inds = os.listdir(path_dico)
            for j in range(len(inds)):
                num_ind = inds[j]
                path_recons = path_dico+num_ind
                ref = np.load(os.path.join(path_recons, 'ref.npy'))
                a, b, mask = _preprocess_input(ref, ref, mask='auto')
                scipy.io.savemat(os.path.join(path_recons, 'mask.mat'),
                                 mdict={'mask': mask})
                np.save(os.path.join(path_recons, 'mask.npy', mask))
                imsave(os.path.join(path_recons, 'mask.png'), np.abs(mask))
                f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
                ax1.imshow(abs(ref), cmap='gray')
                ax2.imshow(mask, cmap='gray')
    if verbose:
        print('[info] compute_mask successfully ended!')

def create_imglist_from_gridsearchresults(CV_path, verbose=False):
    """ create image list of reconstruction from the grid_search results
        pkl object in the corresponding folder

    Parameters:
    -----
        CV_path: str, path to the crossvalidation folder
    Return:
    -------
        str, massage to indicate that the algorithm successfully ended
    """
    nb_cv_iter = len(os.listdir(CV_path))
    for i in range(1, nb_cv_iter + 1):
        if verbose:
            print("[info] crossval iter number: {0}".format(i))
        path = os.path.join(CV_path, 'iter_'+str(i), 'dicos')
        dicos = os.listdir(path)
        for k in range(len(dicos)):
            dico_name = dicos[k]
            path_dico = os.path.join(CV_path, 'iter_'+str(i), 'dicos',
                                    dico_name, 'reconstructions')
            inds = os.listdir(path_dico)
            for j in range(len(inds)):
                num_ind = inds[j]
                path_recons = path_dico + num_ind
                res_gridsearch = load_object(path_recons + '/res.pkl')
                imgs = []
                for l in range(len(res_gridsearch)):
                    imgs.append(res_gridsearch[l][0].data)
                scipy.io.savemat(os.path.join(path_recons,
                                             'imgs_gridsearch.mat'),
                                 mdict={'imgs_gridsearch': imgs})
    if verbose:
        print('[info] create_imglist_from_gridsearchresults successfully'
              'ended!')


def save_best_pisap_recons(CV_path, verbose):
    """ save best pisap reconstruction and dual solution

    Parameters:
    -----
        CV_path: str, path to the crossvalidation folder
    Return:
    -------
        str, massage to indicate that the algorithm successfully ended
    """
    nb_cv_iter = len(os.listdir(CV_path)) -1
    for i in range(1, nb_cv_iter + 1):
        if verbose:
            print("[info] crossval iter number: {0}".format(i))
        path = os.path.join(CV_path, 'iter_'+str(i), 'dicos')
        dicos = os.listdir(path)
        for k in range(len(dicos)):
            dico_name = dicos[k]
            path_dico = os.path.join(CV_path, 'iter_'+str(i), 'dicos',
                                     dico_name, 'reconstructions')
            inds = os.listdir(path_dico)
            for j in range(len(inds)):
                num_ind = inds[j]
                path_recons = path_dico + num_ind
                # reconstructed image
                res = load_object(os.path.join(path_recons, 'res.pkl'))
                best_recons_pisap = scipy.io.loadmat(os.path.join(path_recons,
                                                    'best_recons_pisap.mat'))
                best_recons = best_recons_pisap['reconstruction']
                ind_max = best_recons_pisap['ind_max'][0][0]
                x = res[ind_max][0]
                scipy.io.savemat(path_recons+'/pisap_recons.mat',
                                 mdict={'pisap_recons': x.data})
                imsave(os.path.join(path_recons, 'pisap_recons.png'),
                                    min_max_normalize((np.abs(x.data))))
                # dual_solution
                y = res[ind_max][1]
                scipy.io.savemat(os.path.join(path_recons, 'dual_solution.mat'),
                                 mdict={'dual_solution': y.adj_op(y.coeff)})
                imsave(os.path.join(path_recons, 'dual_solution.png'),
                    min_max_normalize(np.abs(y.adj_op(y.coeff))))
                # results synthesis
                A = load_object(os.path.join(path_recons, 'list_kwargs.pkl'))
                mu_grid = []
                for m in range(len(A)):
                    mu_grid.append(A[m]['mu'])
                mu_on_border = False
                if ind_max == 0 or ind_max == len(mu_grid):
                    mu_on_border = True
                early_stopping = A[ind_max]['metrics']['ssim']['early_stopping']
                nb_iters = A[ind_max]['metrics']['ssim']['index'][-1]
                elapsed_time = A[ind_max]['metrics']['ssim']['time'][-1]
                synthesis_infos_res = {'mu': mu_grid[ind_max],
                                   'mu_on_border': mu_on_border,
                                   # nb d'iteration a la periode d'appel des
                                   # metriques pres:
                                   'nb_iters': nb_iters,
                                   'early_stopping': early_stopping,
                                   # temps ecoule a la periode d'appel des
                                   # metriques pres:
                                   'elapsed_time':elapsed_time,
                                  }
                save_object(synthesis_infos_res, os.path.join(path_recons,
                    'synthesis_info_res.pkl'))
    if verbose:
        print('[info] save_best_pisap_recons successfully ended!')


def generate_flat_patches(images_training_set, patch_size, option='real'):
    """Learn the dictionary from the real/imaginary part or the module/phase
    of images from the training set

    Parameters
    ----------
        images_training_set -- dico of list of 2d matrix of complex values,
            one list per subject
        patch_size -- int, width of square patches
        option -- 'real' (default), 'imag' real/imaginary part or module/phase,
            'complex'
    Return
    ------
        flat_patches -- list of 1d array of flat patches (floats)
    """
    patch_shape = (patch_size, patch_size)
    flat_patches = images_training_set[:]
    for imgs in flat_patches:
        flat_patches_sub = []
        for img in imgs:
            if option == 'real':
                image = np.real(img)
            if option == 'imag':
                image = np.imag(img)
            if option == 'complex':
                image = img
            patches = extract_patches_from_2d_images(min_max_normalize(image),
                                                     patch_shape)
            flat_patches_sub.append(patches)
        yield flat_patches_sub


def reconstruct_2d_images_from_flat_patched_images(flat_patches_list, dico,
                                                   img_shape, threshold=1):
    """ Return the list of 2d reconstructed images from sparse code
        (flat patches and dico)

    Parameters
    ----------
        flat_patches_list: list of 2d np.ndarray of size nb_patches*len_patches
        dico: sklearn MiniBatchDictionaryLearning object
        img_shape: tuple of int, image shape
        threshold (default =1): thresholding level of the sparse coefficents

    Return
    ------
        reconstructed_images: list of 2d np.ndarray of size w*h
    """
    reconstructed_images = []
    patch_size = int(np.sqrt(flat_patches_list[0].shape[1]))
    for i in range(len(flat_patches_list)):
        loadings = dico.transform(flat_patches_list[i])
        norm_loadings = min_max_normalize(np.abs(loadings))
        if threshold > 0:
            loadings[norm_loadings < threshold] = 0
        recons = np.dot(loadings, dico.components_)
        recons = recons.reshape((recons.shape[0], patch_size, patch_size))
        recons = reconstruct_from_patches_2d(recons, img_shape)
        reconstructed_images.append(recons)
    return reconstructed_images
