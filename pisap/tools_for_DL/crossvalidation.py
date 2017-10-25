""" Crossvalidation """

# imports
import numpy as np
import matplotlib.pyplot as plt
import pisap
from pisap.tools_for_DL.images import LisaImages,  LisaImages_forCV
from pisap.tools_for_DL.dictionary import Dictionary
from pisap.numerics.gradient import Grad2DAnalysis, Grad2DSynthesis
from pisap.numerics.linear import Wavelet, DictionaryLearningWavelet, DictionaryLearningWavelet_complex
from pisap.numerics.fourier import NFFT, FFT
from pisap.numerics.cost import ssim, snr, psnr, nrmse
from pisap.base.utils import subsampling_op, min_max_normalize
from pisap.base.utils import convert_mask_to_locations, convert_locations_to_mask, crop_sampling_scheme
from pisap.numerics.reconstruct import sparse_rec_condat_vu
from pisap.base.utils import extract_patches_from_2d_images
from pisap.numerics.gridsearch import grid_search
import data_provider
from sklearn.model_selection import train_test_split
import datetime


def crossvalidation_Lisa(crossval_iter, mu_grid, main_metric, nb_atoms_grid, patch_size_grid, alpha_grid):
    """ Compute crossvalidation to set the dictionary hyperparameters.

    Parameters
    ----------
    crossval_iter: int, the number of crossvalidation loops
    pisap_parameters_gridsearch: dictionary of parameters with a list of mu values to test
    main_metric (default='ssim'), can be 'psnr','snr','nrmse', reference metric 
    nb_atoms_grid: number of atoms to test
    patch_size_grid: patch size to test 
    alpha_grid: dictionary regularisation parameter to test
    Returns
    -------
    res: nd.nparray with column names: nb_atoms, patch_shape, alpha, mu, 
        mean_ssim, mean_psnr, mean_snr, mean_nrmse
    """
    #date=datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    #crossval_dir='CV'+date
    #os.makedirs(crossval_dir)
    res={'nb_atoms':[], 'patch_size':[], 'alpha':[], 'mu':[], 'mean_ssim':[],
    'mean_psnr':[],'mean_snr':[], 'mean_nrmse':[]}
    path="Lisa_complex_all/"
    refA=[]
    
    metrics = {'ssim':{'metric':ssim,
                      'mapping': {'x_new': 'test', 'y_new':None},
                      'cst_kwargs':{'ref':refA},
                      'early_stopping': True,
                       },
               'snr':{'metric':snr,
                      'mapping': {'x_new': 'test', 'y_new':None},
                      'cst_kwargs':{'ref':refA},
                      'early_stopping': True,
                       },
               'psnr':{'metric':psnr,
                      'mapping': {'x_new': 'test', 'y_new':None},
                      'cst_kwargs':{'ref':refA},
                      'early_stopping': True,
                       }, 
               'nrmse':{'metric':nrmse,
                      'mapping': {'x_new': 'test', 'y_new':None},
                      'cst_kwargs':{'ref':refA},
                      'early_stopping': True,
                       },           
              }
    sampling_scheme_path='sampling_schemes/scheme_256_R5_power1_fullCenter.mat'
    for patch_size in patch_size_grid:
        images=LisaImages_forCV(path, patch_size)
        images.load_imgs()
        # getting and plotting the sampling scheme
        sampling_scheme=images.get_sampling_scheme(sampling_scheme_path)
        sampling_scheme = sampling_scheme.astype('float')
        samples = convert_mask_to_locations(sampling_scheme)
        ft_obj = NFFT(samples_locations=samples, img_shape = images.img_shape)
        gradient_param = {"ft_cls": {NFFT: {'samples_locations': samples, 'img_shape':images.img_shape}}}
        for nb_atoms in nb_atoms_grid:
            for alpha in alpha_grid:
                res['nb_atoms']= nb_atoms
                res['patch_size']= patch_size
                res['alpha']= alpha
                print(nb_atoms, patch_size, alpha)
                
                for i in range(crossval_iter):
                    #os.makedirs(crossval_dir+'/iter_'+i)
                    #os.makedirs(crossval_dir+'/iter_'+i+'/refs_train')
                    #os.makedirs(crossval_dir+'/iter_'+i+'/refs_val')
                    #os.makedirs(crossval_dir+'/iter_'+i+'/dicos')
                    #os.makedirs(crossval_dir+'/iter_'+i+'/refs_val')
                    refs_train=images.imgs[:len(images.imgs)-10]
                    refs_val=images.imgs[len(images.imgs)-10:]
                    #save ref
                    images.extract_patches_and_flatten_imgs_trainingset(refs_train)
                    dictionary_real=Dictionary(nb_atoms, alpha, 100)
                    dictionary_imag=Dictionary(nb_atoms, alpha, 100)
                    dictionary_real.learning_atoms(images,'real')
                    dictionary_imag.learning_atoms(images,'imaginary')
                    #np.save(crossval_dir+'/iter_'+i+'/dico_real.npy)
                    func=sparse_rec_condat_vu
                    for j in range(len(refs_val)):
                        # np.save(crossval_dir+'/iter_'+i+'/refs_val/ref_val_'+j+'.npy', refs_val[j])
                        refA=np.abs(refs_val[j])
                        data_undersampled= ft_obj.op(np.abs(refs_val[j]))
                        DLW_r=DictionaryLearningWavelet(dictionary_real.dico,dictionary_real.dico.components_,images.img_shape)
                        DLW_i=DictionaryLearningWavelet(dictionary_imag.dico,dictionary_imag.dico.components_,images.img_shape)
                        pisap_parameters_gridsearch={'data':data_undersampled,
                            'gradient_cls':Grad2DAnalysis,
                            'gradient_kwargs':gradient_param,
                            'linear_cls':DictionaryLearningWavelet_complex,
                            'linear_kwargs':{"DLW_r": DLW_r, "DLW_i": DLW_i},
                            'max_nb_of_iter':100,
                            'add_positivity': False,
                            'mu':mu_grid,
                            'metrics':metrics,
                            'verbose':1,
                             }
                        Gridsearch=grid_search(func,pisap_parameters_gridsearch,verbose=1)
                        #save pisap parameters
        
                        max_metric=[]
                        nb_mu=len(pisap_parameters_gridsearch['mu'])
                        for i in range(nb_mu):
                             metric_values=Gridsearch[1][i][2][main_metric]['values']
                        if main_metric=='nrmse':
                            metric_val=min(metric_values)
                        else:
                            metric_val=max(metric_values)
                        max_metric.append(metric_val)
                    if main_metric=='nrmse':
                        ind_max=max_metric.index(min(max_metric))
                    else:
                        ind_max=max_metric.index(max(max_metric))
                    
    return(res)
                            
        

            