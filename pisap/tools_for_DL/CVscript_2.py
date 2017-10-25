################################################ IMPORTS #######################################################

# basic imports
import matplotlib.pyplot as plt
import scipy.fftpack as pfft
import numpy as np
import scipy.io

#pisap imports
import pisap
print pisap.__version__

from pisap.numerics.gradient import Grad2DAnalysis, Grad2DSynthesis
from pisap.numerics.linear import Wavelet, DictionaryLearningWavelet, DictionaryLearningWavelet_complex
from pisap.numerics.fourier import NFFT, FFT
from pisap.numerics.cost import ssim, snr, psnr, nrmse
from pisap.base.utils import subsampling_op, min_max_normalize
from pisap.base.utils import convert_mask_to_locations, convert_locations_to_mask, crop_sampling_scheme
from pisap.numerics.reconstruct import sparse_rec_condat_vu
from pisap.numerics.gridsearch import grid_search
import data_provider

from pisap.tools_for_DL.images import LisaImages, LisaImages_forCV
from pisap.tools_for_DL.dictionary import Dictionary
from pisap.tools_for_DL.crossvalidation import crossvalidation_Lisa

from sklearn.feature_extraction.image import extract_patches_2d
#from pisap.tools_for_DL.images import read_dicom_img, save_all_dicom_img
from sklearn.decomposition import MiniBatchDictionaryLearning

# imports
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import datetime
import os
import pickle
from sklearn.model_selection import ShuffleSplit
from random import shuffle
from scipy.misc import imsave
import random

################################################## PARAMETERS ###########################################################

size_validation_set=10
crossval_iter=1

mu_grid=[1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,5e-2,1e-1,0.5,1,2,5]

################################################# CROSSVALIDATION ########################################################

#paths
path="Lisa_training_validation_sets/"
sampling_scheme_path='sampling_schemes/scheme_256_R5_power1_fullCenter.mat'

# create folder
date=datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
crossval_dir='/home/hc253658/STAGE/Crossvalidations/CV_'+date
os.makedirs(crossval_dir)

#parameters and initialization
func=sparse_rec_condat_vu
results={'nb_atoms':[],
         'patch_size':[],
         'alpha':[],
         'mu':[],
         'ssim':[],
         'psnr':[]
        }

#create subfolder iter, subsubfolder refs_train and dicos
title_iter=crossval_dir+'/iter_'+str(1)
os.makedirs(title_iter)
os.makedirs(title_iter+'/refs_train')
title_dico=title_iter+'/dicos'
os.makedirs(title_dico)

#faire une liste dir des dicos, extraire infos nb atoms, pathc_size, ...
dicos_path='/home/hc253658/STAGE/Crossvalidations/CV_2017-10-20_19:28:12'
list_dicos=os.listdir(dicos_path)

for dico_name in list_dicos:
    nb_atoms=int(dico_name[14:17])
    patch_size=int(dico_name[6])
    alpha=0.1

    #create instance of class Lisaimages_forCV
    images=LisaImages_forCV(path, patch_size)
    images.load_imgs()

    sampling_scheme=images.get_sampling_scheme(sampling_scheme_path)
    sampling_scheme = sampling_scheme.astype('float')
    imsave(crossval_dir+'/sampling_scheme.png', sampling_scheme)
    samples = convert_mask_to_locations(pfft.fftshift(sampling_scheme))
    ft_obj = FFT(samples_locations=samples, img_shape = images.img_shape)
    gradient_param = {"ft_cls": {FFT: {'samples_locations': samples, 'img_shape':images.img_shape}}}

    print('nb_atoms',nb_atoms,'patch_size',patch_size,'alpha',alpha)
    #create subfolders for the dictionary, and the reconstructions
    dico_folder=title_dico+'/patch_'+str(patch_size)+'_atoms_'+str(nb_atoms)+'_alpha_'+str(alpha)
    os.makedirs(dico_folder)
    os.makedirs(dico_folder+'/reconstructions')

    #index testing set not in training set and shuffle
    # provide shuffle training and test set with index
    index=[i for i in range(len(images.imgs))]
    index_train=np.load('/home/hc253658/STAGE/Crossvalidations/CV_2017-10-20_19:28:12.npy')
    index_aux=index
    for i in range(len(index_train)):
        index.remove(index_train[i])
    index_val=[]
    while len(index_val<10):
        ind=random.choice(index_aux)
        if ind not in index_val:
            index_val.append(ind)
    refs_train=[images.imgs[index_train[i]] for i in range(len(index_train))]
    refs_val=[images.imgs[index_val[i]] for i in range(len(index_val))]

    #save refs_train
    for i in range(len(refs_train)):
        np.save(title_iter+'/refs_train/ref_train_'+str(index_train[i])+'.npy', refs_train[i])
        imsave(title_iter+'/refs_train/ref_train_'+str(index_train[i])+'.png', min_max_normalize(np.abs(refs_train[i])))

    #preprocessing data, learning and saving dictionaries
    images.extract_patches_and_flatten_imgs_trainingset(refs_train)
    f=open(dicos_path+'/dico_real.p','w')
    dictionary_real=pickle.load(f)
    f.close()
    f=open(dicos_path+'/dico_imag.p','w')
    dictionary_imag=pickle.load(f)
    f.close()

    for j in range(len(refs_val)): # for each image of the validation set

        #create subfolders and save reference
        ref=np.abs(refs_val[j])
        path_reconstruction=dico_folder+'/reconstructions/ind_'+str(index_val[j])
        os.makedirs(path_reconstruction)
        np.save(path_reconstruction+'/ref.npy', refs_val[j])
        imsave(path_reconstruction+'/ref.png', min_max_normalize(np.abs(refs_val[j])))

        #metrics with reference
        metrics = {'ssim':{'metric':ssim,
                          'mapping': {'x_new': 'test', 'y_new':None},
                          'cst_kwargs':{'ref':ref},
                          'early_stopping': True,
                           },
                   'snr':{'metric':snr,
                          'mapping': {'x_new': 'test', 'y_new':None},
                          'cst_kwargs':{'ref':ref},
                          'early_stopping': True,
                           },
                   'psnr':{'metric':psnr,
                          'mapping': {'x_new': 'test', 'y_new':None},
                          'cst_kwargs':{'ref':ref},
                          'early_stopping': True,
                           },
                   'nrmse':{'metric':nrmse,
                          'mapping': {'x_new': 'test', 'y_new':None},
                          'cst_kwargs':{'ref':ref},
                          'early_stopping': True,
                           },
                  }

        #settle pisap gridsearch parameters
        data_undersampled= ft_obj.op(np.abs(refs_val[j]))
        DLW_r=DictionaryLearningWavelet(dictionary_real.dico,dictionary_real.dico.components_,images.img_shape)
        DLW_i=DictionaryLearningWavelet(dictionary_imag.dico,dictionary_imag.dico.components_,images.img_shape)

        pisap_parameters_gridsearch={'data':data_undersampled,
            'gradient_cls':Grad2DAnalysis,
            'gradient_kwargs':gradient_param,
            'linear_cls':DictionaryLearningWavelet_complex,
            'linear_kwargs':{"DLW_r": DLW_r, "DLW_i": DLW_i},
            'max_nb_of_iter':nb_max_iter,
            'add_positivity': False,
            'mu':mu_grid,
            'metrics':metrics,
            'verbose':1,
             }

        #computing gridsearch
        gridsearch=grid_search(func,pisap_parameters_gridsearch,n_jobs=len(mu_grid),verbose=1)

        #choose best reconstruction, SSIM criteron
        ssims=[]
        psnrs=[]
        for i in range(len(mu_grid)):
            ssim_values=gridsearch[1][i][2]['ssim']['values']
            pnsr_values=gridsearch[1][i][2]['psnr']['values']
            ssim_val=max(ssim_values)
            psnr_val=max(pnsr_values)
            ssims.append(ssim_val)
            psnrs.append(psnr_val)
        max_ssim=max(ssims)
        ind_max=ssims.index(max(ssims))
        psnr_ind_max=psnrs[ind_max]
        mu=mu_grid[ind_max]

        #results
        results['nb_atoms'].append(nb_atoms)
        results['alpha'].append(alpha)
        results['patch_size'].append(patch_size)
        results['mu'].append(mu)
        results['ssim'].append(max_ssim)
        results['psnr'].append(psnr_ind_max)

        #final recosntruction
        pisap_parameters={'data':data_undersampled,
            'gradient_cls':Grad2DAnalysis,
            'gradient_kwargs':gradient_param,
            'linear_cls':DictionaryLearningWavelet_complex,
            'linear_kwargs':{"DLW_r": DLW_r, "DLW_i": DLW_i},
            'max_nb_of_iter':nb_max_iter,
            'add_positivity': False,
            'mu':mu,
            'metrics':metrics,
            'verbose':1,
             }

        x, y, saved_metric = sparse_rec_condat_vu(**pisap_parameters)

        #reconstruction
        np.save(path_reconstruction+'/reconstruction.npy', x.data)
        imsave(path_reconstruction+'/reconstruction.png', min_max_normalize((np.abs(x.data))))
        #dual_solution
        np.save(path_reconstruction+'/dual_solution.npy', y.adj_op(y.coeff))
        imsave(path_reconstruction+'/dual_solution.png', min_max_normalize(np.abs(y.adj_op(y.coeff))))
        #zero-order solution
        np.save(path_reconstruction+'/zero_order_solution.npy', ft_obj.adj_op(data_undersampled))
        imsave(path_reconstruction+'/zero_order_solution.png', min_max_normalize(np.abs(ft_obj.adj_op(data_undersampled))))
        #save parameters
        f = open(path_reconstruction+'/pisap_param_final_reconstruction.txt',"w")
        f.write( str(pisap_parameters) )
        f.close()
        f = open(path_reconstruction+'/saved_metrics_final_reconstruction.txt',"w")
        f.write( str(saved_metric) )
        f.close()
        f = open(path_reconstruction+'/gridsearch.txt',"w")
        f.write( str(gridsearch) )
        f.close()
        #save control
        early_stopping=True
        if gridsearch[1][0][2]['ssim']['index'][-1]==nb_max_iter:
            early_stopping=False
        mu_on_border=False
        if ind_max==0 or ind_max==len(mu_grid):
            mu_on_border=True
        control={'early_stopping':early_stopping,'mu_on_border':mu_on_border}
        f = open(path_reconstruction+'/control.txt',"w")
        f.write( str(control) )
        f.close()

    f = open(crossval_dir+'/results.txt',"w")
    f.write( str(results) )
    f.close()


    print(results)
