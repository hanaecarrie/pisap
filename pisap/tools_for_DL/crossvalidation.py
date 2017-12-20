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
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from joblib import Parallel, delayed

# Package import
from pisap.base.utils import min_max_normalize
from pisap.tools_for_DL.pickle import load_object, save_object
from pisap.base.utils import extract_patches_from_2d_images
from pisap.base.utils import timer
from pisap.numerics.linear import DictionaryLearningWavelet
from pisap.numerics.fourier import NFFT, FFT
from pisap.numerics.gridsearch import grid_search
from pisap.numerics.cost import _preprocess_input


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
        os.makedirs(os.path.join(title_iter,'refs_train'))
        os.makedirs(os.path.join(title_iter,'refs_val'))
        title_dico = os.path.join(title_iter, 'dicos')
        os.makedirs(title_dico)

        num_subject_val = random.randint(1, nb_subjects)
        print(num_subject_val)
        training_set = imgs[:]
        training_set = training_set.tolist()
        del training_set[num_subject_val-1]
        np.save(os.path.join(title_iter,'refs_train','training_set.npy'),
                             training_set)
        validation_set = imgs[num_subject_val-1]
        np.save(os.path.join(title_iter,'refs_val','validation_set.npy'),
                             validation_set)
        save_object(num_subject_val, os.path.join(title_iter,'refs_val',
                    'num_subject_val.pkl'))

        # provide training set
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
            flat_patches_real = generate_flat_patches(imgs, patch_size,
                                                      'real')
            dico_real = generate_dico(flat_patches_real, nb_atoms, alpha,
                                      n_iter_dico,
                                      fit_algorithm=fit_algorithm,
                                      transform_algorithm=transform_algorithm,
                                      batch_size=batch_size)
            save_object(dico_real, dico_folder+'/dico_real.pkl')
            flat_patches_imag = generate_flat_patches(imgs, patch_size,
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
        num_subject_val = load_object(os.path.join(CV_path,
                                 'iter_'+ str(i), 'refs_val',
                                 'num_subject_val.pkl'))
        validation_set = np.load(os.path.join(CV_path,
                                 'iter_'+ str(i), 'refs_val', 'validation_set.npy'))
        validation_set = validation_set[:10]
        img_shape = np.array(validation_set[0]).shape

        for dico in dicos_list:
            if verbose:
                print('[info] Dictionary parameters:{0}'.format(dico))
            dico_folder = os.path.join(CV_path, 'iter_'+str(i), 'dicos', dico)
            os.makedirs(os.path.join(CV_path, 'iter_'+str(i), 'dicos', dico,
                                     'reconstructions'))
            dico_real = load_object(os.path.join(dico_folder, 'dico_real.pkl'))
            dico_imag = load_object(os.path.join(dico_folder, 'dico_imag.pkl'))
            patch_size = int(np.sqrt(dico_real.components_.shape[1]))
            t0 = time.time()
            flat_patches_real = generate_flat_patches_2(validation_set,
                                                      patch_size, 'real')
            flat_patches_imag = generate_flat_patches_2(validation_set,
                                                      patch_size, 'imag')
            duration = time.time() -t0
            if verbose:
                print('[info] Generate patches in: {0}'.format(duration))

            for j in range(len(validation_set)):
                if verbose:
                    print('[info] Image number: {0}'.format(j))
                t1 = time.time()
                loadings_r = dico_real.transform(flat_patches_real[j])
                norm_loadings_r = min_max_normalize(np.abs(loadings_r))
                if threshold > 0:
                    loadings_r[norm_loadings_r < threshold] = 0
                recons_r = np.dot(loadings_r, dico_real.components_)
                recons_r = recons_r.reshape((recons_r.shape[0],
                           patch_size, patch_size))
                recons_r = reconstruct_from_patches_2d(recons_r, img_shape)
                duration_recons_r = time.time() -t1
                if verbose:
                    print('[info] Recons real in: {0}'.format(duration_recons_r))
                t2 = time.time()
                loadings_i = dico_imag.transform(flat_patches_imag[j])
                norm_loadings_i = min_max_normalize(np.abs(loadings_i))
                if threshold > 0:
                    loadings_i[norm_loadings_i < threshold] = 0
                recons_i = np.dot(loadings_r, dico_imag.components_)
                recons_i = recons_i.reshape((recons_i.shape[0],
                           patch_size, patch_size))
                recons_i = reconstruct_from_patches_2d(recons_i, img_shape)
                duration_recons_i = time.time() -t2
                if verbose:
                    print('[info] Recons imag in: {0}'.format(duration_recons_i))

                recons = recons_r[j]+1j*recons_i[j]
                #saving_reference
                path_reconstruction = os.path.join(dico_folder,
                    'reconstructions', 'ind_'+str(j))
                os.makedirs(path_reconstruction)
                scipy.io.savemat(os.path.join(path_reconstruction, 'ref.mat'),
                                 mdict={'ref': validation_set[j]})
                imsave(os.path.join(path_reconstruction, 'ref.png'),
                       min_max_normalize(np.abs(validation_set[j])))
                #reconstructed image
                scipy.io.savemat(os.path.join(path_reconstruction,
                                 'sparse_code_recons.mat'),
                                 mdict={'sparse_code_recons': recons})
                imsave(os.path.join(path_reconstruction,
                                    'sparse_code_recons.png'),
                                    min_max_normalize((np.abs(recons))))
    if verbose:
        print('[info] compute_sparse_code_forCV successfully ended!')


def compute_pisap_gridsearch_forCV(CV_path, sampling_scheme, func,
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
    nb_cv_iter = len(os.listdir(CV_path))
    is_first_iter = True

    #for i in range(1, nb_cv_iter + 1):
    i = 1
    if verbose:
        print("[info] crossval iteration: {0}".format(i))
    dicos_list = os.listdir(os.path.join(CV_path, 'iter_'+str(i), 'dicos'))
    validation_set = np.load(os.path.join(CV_path,
                             'iter_'+ str(i), 'refs_val/validation_set.npy'))
    validation_set = validation_set[:5]
    img_shape = np.array(validation_set[0]).shape
    kspaceCS = []
    for val in validation_set:
        if Fourier_option == FFT:
            kspaceCS_i = np.fft.fft2(val)*pfft.fftshift(sampling_scheme)
        elif Fourier_option == NFFT:
            kspaceCS_i = np.fft.fft2(val)*(sampling_scheme)
        kspaceCS_i = pfft.fftshift(kspaceCS_i)
    kspaceCS.append(kspaceCS_i)
    img_shape = np.array(validation_set[0]).shape
    #index_val = os.listdir(os.listdir(CV_path, 'iter_'+str(i)))[1]
    #params_dico = load_object(os.path.join(CV_path, 'iter_'+str(i), 'dicos'))
    Parallel(n_jobs=30, verbose=11)(
    delayed(_run_grid_search)(dico, CV_path, i, is_first_iter,
                              pisap_parameters_grid, validation_set,
                              func, ft_obj, verbose)
                                    for dico in dicos_list)


def _run_grid_search(dico, CV_path, i, is_first_iter, pisap_parameters_grid,
                     validation_set, func, ft_obj, verbose):
    """
    """
    if verbose:
        print('[info] Dictionary parameters: {0}'.format(dico))
    dico_folder = os.path.join(CV_path, 'iter_'+str(i), 'dicos', dico)
    dico_real = load_object(os.path.join(dico_folder, 'dico_real.pkl'))
    dico_imag = load_object(os.path.join(dico_folder, 'dico_imag.pkl'))

    #XXX by default the pickled dico launch -1 jobs to 'transform'
    dico_real.n_jobs = 1
    dico_imag.n_jobs = 1

    #list_args = load_object(os.path.join(dico_folder,
    #                                    'dico_parameters.pkl'))
    # patch_size = list_args['patch_size']
    patch_size = int(np.sqrt(dico_real.components_.shape[1]))
    if is_first_iter: #TODO debug this
        pisap_parameters_grid['linear_kwargs']['dictionary_r'] = dico_real
        pisap_parameters_grid['linear_kwargs']['dictionary_i'] = dico_imag
    else:
        pisap_parameters_grid['linear_kwargs'][0]['dictionary_r'] = dico_real
        pisap_parameters_grid['linear_kwargs'][0]['dictionary_i'] = dico_imag
    if os.path.isdir(os.path.join(dico_folder, 'reconstructions'))==False:
        os.makedirs(os.path.join(dico_folder, 'reconstructions'))
    for j in range(len(validation_set)):
        if verbose:
            print("[info] index of current validation image:"
                  "{0}".format(j))
        path_reconstruction = os.path.join(dico_folder, 'reconstructions',
                                           'ind_'+str(j+1))
        if os.path.isdir(path_reconstruction)==False:
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
                                       n_jobs=1,
                                       verbose=11)
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
                path_recons = os.path.join(path_dico, num_ind)
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
