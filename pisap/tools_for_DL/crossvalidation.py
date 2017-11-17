""" Crossvalidation """

# imports

# basic imports
import matplotlib.pyplot as plt
import scipy.fftpack as pfft
import numpy as np
import scipy.io
import datetime
import os
import pickle
from scipy.misc import imsave
from random import shuffle
import itertools
#pisap imports
import pisap
from pisap.base.utils import min_max_normalize
from pisap.base.utils import  generate_flat_patches, generate_dico, reconstruct_2d_images_from_flat_patched_images
from pisap.base.utils import save_object, load_object
# sklearn imports
import sklearn
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import MiniBatchDictionaryLearning

def generate_dicos_for_CV(CV_params, IMGS_params, DICOS_params):
    """ Learn dicos from the IMGS over a grid of parameters
        Create folders and files to save the learnt dicos

    Parameters:
    -----------
        CV_params: dictionary of parameters
            example:    CV_params={'nb_cv_iter':1,
                        'saving_path':'/home/hc253658/STAGE/Crossvalidations/CV_',
                        'size_validation_set':2}
        IMGS_params: dictionary of parameters
            example:    IMGS_params={'imgs':imgs[:10],
                        'img_shape':imgs[0].shape
                        }
        DICOS_params: dictionary of parameters
            example:    DICOS_params={'param_grid':{'nb_atoms':[200,300],
                                                    'patch_size':[2],
                                                    'alpha':[1e-4]
                                                     },
                                      'n_iter_dico':10}
    Return:
    -------
        str, massage to indicate that the algorithm successfully ended
    """
    #parameters
    nb_cv_iter = CV_params['nb_cv_iter']
    saving_path = CV_params['saving_path']
    size_validation_set = CV_params['size_validation_set']
    imgs = IMGS_params['imgs']
    img_shape = IMGS_params['img_shape']
    nb_atoms = DICOS_params['param_grid']['nb_atoms']
    patch_size = DICOS_params['param_grid']['patch_size']
    alpha = DICOS_params['param_grid']['alpha']
    n_iter_dico = DICOS_params['n_iter_dico']

    # create folder
    date=datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    crossval_dir=saving_path+date
    os.makedirs(crossval_dir)

    list_args = [dict(zip(DICOS_params['param_grid'], x))
                   for x in itertools.product(*DICOS_params['param_grid'].values())]


    for i in range(1,nb_cv_iter+1):

        #create subfolder iter, subsubfolder refs_train and dicos
        title_iter=crossval_dir+'/iter_'+str(i)
        os.makedirs(title_iter)
        os.makedirs(title_iter+'/refs_train')
        os.makedirs(title_iter+'/refs_val')
        title_dico=title_iter+'/dicos'
        os.makedirs(title_dico)

        #provide shuffle training and test set with index
        index=[i for i in range(len(imgs))]
        shuffle(index)
        index_train=index[size_validation_set:]
        index_val=index[:size_validation_set]
        training_set=[imgs[i] for i in index_train]
        validation_set=[imgs[i] for i in index_val]

        #save training_set
        for i in range(len(training_set)):
            np.save(title_iter+'/refs_train/ref_train_'+str(index_train[i])+'.npy', training_set[i])
            imsave(title_iter+'/refs_train/ref_train_'+str(index_train[i])+'.png', \
                   min_max_normalize(np.abs(training_set[i])))

        #save validation_set
        np.save(title_iter+'/refs_val/refs_val.npy', validation_set)
        np.save(title_iter+'/refs_val/index_val.npy', index_val)

        #testing different dictionary parameters
        for args in list_args:

            #print dico parameters
            print(args)
            for key,val in args.items():
                exec(key + '=val')

            #create subfolders for the dictionary, and the reconstructions
            dico_folder=title_dico+'/patch='+str(patch_size)+'_atoms='+str(nb_atoms)+'_alpha='+str(alpha)
            os.makedirs(dico_folder)
            os.makedirs(dico_folder+'/reconstructions')

            #preprocessing data, learning and saving dictionaries
            flat_patches_real=generate_flat_patches(training_set,patch_size, 'real')
            dico_real=generate_dico(flat_patches_real, nb_atoms, alpha, n_iter_dico)
            flat_patches_imag=generate_flat_patches(training_set,patch_size, 'imag')
            dico_imag=generate_dico(flat_patches_imag, nb_atoms, alpha, n_iter_dico)
            np.save(dico_folder+'/dico_real.npy', dico_real.components_)
            np.save(dico_folder+'/dico_imag.npy', dico_imag.components_)
            ### save pickle dictionary
            save_object(dico_real, dico_folder+'/dico_real.pkl')
            save_object(dico_imag, dico_folder+'/dico_imag.pkl')

    return('Generate and save dictionaries successfully ended!')


def compare_dictionaries_SparseCode(CV_path, threshold=0):
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
    nb_cv_iter=len(os.listdir(CV_path+'/'))

    for i in range(1,nb_cv_iter+1):
        dicos_list = os.listdir(CV_path+'/iter_'+str(i)+'/dicos/')
        validation_set=np.load(CV_path+'/iter_'+str(i)+'/refs_val/refs_val.npy')
        validation_set=validation_set.tolist()
        for val in validation_set:
            val=np.array(val)
        img_shape = np.array(validation_set[0]).shape
        index_val=np.load(CV_path+'/iter_'+str(i)+'/refs_val/index_val.npy')

        for dico in dicos_list:
            dico_folder=CV_path+'/iter_'+str(i)+'/dicos/'+dico
            dico_real = load_object(dico_folder+'/dico_real.pkl')
            dico_imag = load_object(dico_folder+'/dico_imag.pkl')
            patch_size=int(dico[6])
            flat_patches_real = generate_flat_patches(validation_set, patch_size, 'real')
            flat_patches_imag = generate_flat_patches(validation_set, patch_size, 'imag')
            recons_real = reconstruct_2d_images_from_flat_patched_images(flat_patches_real,dico_real,img_shape, threshold)
            recons_imag = reconstruct_2d_images_from_flat_patched_images(flat_patches_imag,dico_imag, img_shape, threshold)
            for j in range(len(validation_set)):
                recons = recons_real[j]+1j*recons_imag[j]
                #saving_reference
                path_reconstruction = dico_folder+'/reconstructions/ind_'+str(index_val[j])
                os.makedirs(path_reconstruction)
                np.save(path_reconstruction+'/ref.npy', validation_set[j])
                scipy.io.savemat(path_reconstruction+'/ref.mat',mdict={'ref': validation_set[j]})
                imsave(path_reconstruction+'/ref.png', min_max_normalize(np.abs(validation_set[j])))
                #reconstructed image
                np.save(path_reconstruction+'/reconstruction.npy',recons)
                scipy.io.savemat(path_reconstruction+'/reconstruction.mat',mdict={'reconstruction': recons})
                imsave(path_reconstruction+'/reconstruction.png', min_max_normalize((np.abs(recons))))

    return('compare_SparseCode successfully ended!')


def compare_dictionaries_retrospectiveCSreconstruction(CV_path, params):
    """ Load the previousely learnt dictionaries
        Make a pisap grid_search for retrospective CS reconstruction
        Save results and indicate best results depending on ssim

    Parameters:
    -----
        CV_path: str, path to the crossvalidation folder
        threshold (default =0): thresholding level of the sparse coefficents
    Return:
    -------
        str, massage to indicate that the algorithm successfully ended
    """
    #params to variables
    sampling_scheme = params['sampling_scheme']
    mu_grid = params['pisap_parameters_grid']['mu']
    func = params['func']
    pisap_parameters_grid = params['pisap_parameters_grid']
    Fourier_option = params['Fourier_option']
    ft_obj = params['ft_obj']
    ref_metric = params['ref_metric']

    nb_cv_iter=len(os.listdir(CV_path+'/'))

    for i in range(1,nb_cv_iter+1):
        dicos_list = os.listdir(CV_path+'/iter_'+str(i)+'/dicos/')
        validation_set = np.load(CV_path+'/iter_'+str(i)+'/refs_val/refs_val.npy')
        validation_set = validation_set.tolist()
        kspaceCS=[]

        for val in validation_set:
            val=np.array(val)
            if Fourier_option == FFT:
                kspaceCS_i = np.fft.fft2(validation_set[i])*pfft.fftshift(sampling_scheme)
            elif Fourier_option == NFFT:
                kspaceCS_i = np.fft.fft2(validation_set[i])*(sampling_scheme)
            kspaceCS_i=pfft.fftshift(kspaceCS_i)
            kspaceCS.append(kspaceCS_i)
        img_shape = np.array(validation_set[0]).shape
        index_val=np.load(CV_path+'/iter_'+str(i)+'/refs_val/index_val.npy')

        for dico in dicos_list:
            dico_folder = CV_path+'/iter_'+str(i)+'/dicos/'+dico
            dico_real = load_object(dico_folder+'/dico_real.pkl')
            dico_imag = load_object(dico_folder+'/dico_imag.pkl')
            patch_size = int(dico[6])
            # patch_size < 10 !!!

            DLW_r = DictionaryLearningWavelet(dico_real,dico_real.components_,img_shape)
            DLW_i = DictionaryLearningWavelet(dico_imag,dico_imag.components_,img_shape)
            pisap_parameters_grid['linear_kwargs']['DLW_r'] = DLW_r
            pisap_parameters_grid['linear_kwargs']['DLW_i'] = DLW_i

            for j in range(len(validation_set)):
                path_reconstruction = dico_folder+'/reconstructions/ind_'+str(index_val[j])
                os.makedirs(path_reconstruction)
                ref=np.abs(validation_set[j])
                pisap_parameters_grid['metrics']['ssim']['cst_kwargs']['ref'] = ref
                pisap_parameters_grid['metrics']['snr']['cst_kwargs']['ref'] = ref
                pisap_parameters_grid['metrics']['psnr']['cst_kwargs']['ref'] = ref
                pisap_parameters_grid['metrics']['nrmse']['cst_kwargs']['ref'] = ref
                pisap_parameters_grid['data'] = ft_obj.op(validation_set[j])
                data_undersampled = ft_obj.op(np.abs(validation_set[j]))
                pisap_parameters_grid['data'] = data_undersampled
                #computing gridsearch
                list_kwargs, res = grid_search(func,pisap_parameters_grid,do_not_touch=[],n_jobs=len(mu_grid),verbose=1)
                np.save(path_reconstruction+'/list_kwargs.pkl', np.array(list_kwargs))
                np.save(path_reconstruction+'/res.pkl', np.array(res))
                #choose best reconstruction, SSIM criteron
                ssims = []
                for i in range(len(mu_grid)):
                    ssims.append(res[i][2][ref_metric]['values'][-1])
                ind_max = ssims.index(max(ssims))

                #reconstructed image
                x = res[ind_max][0]
                np.save(path_reconstruction+'/reconstruction.npy', x.data)
                scipy.io.savemat(path_reconstruction+'/reconstruction.mat',mdict={'reconstruction': x.data})
                imsave(path_reconstruction+'/reconstruction.png', min_max_normalize((np.abs(x.data))))
                #dual_solution
                y = res[ind_max][1]
                np.save(path_reconstruction+'/dual_solution.npy', y.adj_op(y.coeff))
                scipy.io.savemat(path_reconstruction+'/dual_solution.mat',mdict={'dual_solution': y.adj_op(y.coeff)})
                imsave(path_reconstruction+'/dual_solution.png', min_max_normalize(np.abs(y.adj_op(y.coeff))))
                #zero-order solution
                np.save(path_reconstruction+'/zero_order_solution.npy', ft_obj.adj_op(data_undersampled))
                scipy.io.savemat(path_reconstruction+'/zero_order_solution.mat',mdict={'zero_order_solution':ft_obj.adj_op(data_undersampled)})
                imsave(path_reconstruction+'/zero_order_solution.png', min_max_normalize(np.abs(ft_obj.adj_op(data_undersampled))))
                #save parameters
                save_object(pisap_parameters_gridsearch, path_reconstruction+'/pisap_param_final_reconstruction.pkl')
                save_object(saved_metric, path_reconstruction+'/saved_metrics_final_reconstruction.pkl',"w")
                #save results
                mu_on_border = False
                if ind_max == 0 or ind_max == len(mu_grid):
                    mu_on_border = True
                results = {'mu':mu_grid[ind_max],
                           'mu_on_border':mu_on_border,
                           'early_stopping': res[ind_max][2][ref_metric]['is_early_stop'],
                           #metrics
                          }
                save_object(control, path_reconstruction+'/control.pkl')
    return('compare_CS reconstruction successfully ended!')
