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
from pisap.numerics.gradient import Grad2DAnalysis, Grad2DSynthesis
from pisap.numerics.linear import Wavelet, DictionaryLearningWavelet, DictionaryLearningWavelet_complex
from pisap.numerics.fourier import NFFT, FFT
from pisap.numerics.cost import ssim, snr, psnr, nrmse
from pisap.base.utils import subsampling_op, min_max_normalize
from pisap.base.utils import convert_mask_to_locations, convert_locations_to_mask, crop_sampling_scheme
from pisap.base.utils import plot_img, generate_flat_patches, generate_dico, plot_dico
from pisap.numerics.reconstruct import sparse_rec_condat_vu
from pisap.numerics.gridsearch import grid_search
import data_provider

# sklearn imports
import sklearn
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

def compare_CS_recons_wrt_SSIM_score(func,pisap_parameters_gridsearch, img, metrics, path_reconstruction):
    #metrics
    ref=np.abs(img)
    metrics['ssim']['cst_kwargs']['ref'] = ref
    metrics['snr']['cst_kwargs']['ref'] = ref
    metrics['psnr']['cst_kwargs']['ref'] = ref
    metrics['nrmse']['cst_kwargs']['ref'] = ref

    #computing gridsearch
    gridsearch = grid_search(func,pisap_parameters_gridsearch,do_not_touch=[],n_jobs=len(mu_grid),verbose=1)

    #choose best reconstruction, SSIM criteron
    ssims = []
    psnrs = []
    for i in range(len(mu_grid)):
        ssim_values = gridsearch[1][i][2]['ssim']['values']
        pnsr_values = gridsearch[1][i][2]['psnr']['values']
        ssim_val = max(ssim_values)
        psnr_val = max(pnsr_values)
        ssims.append(ssim_val)
        psnrs.append(psnr_val)
    max_ssim = max(ssims)
    ind_max = ssims.index(max(ssims))
    psnr_ind_max = psnrs[ind_max]
    mu = mu_grid[ind_max]

    f = open(path_reconstruction+'/gridsearch.txt',"w")
    f.write( str(gridsearch) )
    f.close()

    return(mu)

def final_CS_reconstruction(func,mu):

    pisap_parameters = pisap_parameters_gridsearch
    pisap_parameters['mu'] = mu
    x, y, saved_metric = func(**pisap_parameters)

    #reconstructed image
    np.save(path_reconstruction+'/reconstruction.npy', x.data)
    scipy.io.savemat(path_reconstruction+'/reconstruction.mat',mdict={'reconstruction': x.data})
    imsave(path_reconstruction+'/reconstruction.png', min_max_normalize((np.abs(x.data))))
    #dual_solution
    np.save(path_reconstruction+'/dual_solution.npy', y.adj_op(y.coeff))
    scipy.io.savemat(path_reconstruction+'/dual_solution.mat',mdict={'dual_solution': y.adj_op(y.coeff)})
    imsave(path_reconstruction+'/dual_solution.png', min_max_normalize(np.abs(y.adj_op(y.coeff))))
    #zero-order solution
    np.save(path_reconstruction+'/zero_order_solution.npy', ft_obj.adj_op(data_undersampled))
    scipy.io.savemat(path_reconstruction+'/zero_order_solution.mat',mdict={'zero_order_solution':ft_obj.adj_op(data_undersampled)})
    imsave(path_reconstruction+'/zero_order_solution.png', min_max_normalize(np.abs(ft_obj.adj_op(data_undersampled))))
    #save parameters
    f = open(path_reconstruction+'/pisap_param_final_reconstruction.txt',"w")
    f.write( str(pisap_parameters) )
    f.close()
    f = open(path_reconstruction+'/saved_metrics_final_reconstruction.txt',"w")
    f.write( str(saved_metric) )
    f.close()
    #save control
    early_stopping = True #XXX problem
    if gridsearch[1][0][2]['ssim']['index'][-1]==nb_max_iter:
        early_stopping = False
        mu_on_border = False
    if ind_max == 0 or ind_max == len(mu_grid):
        mu_on_border = True
    control = {'early_stopping':early_stopping,'mu_on_border':mu_on_border}
    f = open(path_reconstruction+'/control.txt',"w")
    f.write( str(control) )
    f.close()
    results = {'nb_atoms':[],
             'patch_size':[],
             'alpha':[],
             'mu':[],
             'ssim':[],
             'psnr':[]
            }
    return(results)




def read_dicom_img(dicom_path_folder):
    subject=os.listdir(dicom_path_folder)
    for i in range(len(subject)):
        print(i+1)
        dicom_path_sub_folder=dicom_path_folder+subject[i]+'/M/'
        slices=os.listdir(dicom_path_sub_folder)
        for j in range(len(slices)):
            dicom_path_img=dicom_path_sub_folder+slices[j]
            RefDs=dicom.read_file(dicom_path_img)
            ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns))
            ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
            ds = dicom.read_file(dicom_path_img)
            ArrayDicom[:, :] = ds.pixel_array
            ArrayDicom=ArrayDicom.astype('float')
            print(j+1)
            plt.figure()
            plt.imshow(min_max_normalize(ArrayDicom),cmap='gray')
            plt.colorbar()
            plt.axis('equal')
            plt.show()
    return()

def save_dicom(dicom_path_img,title):
    RefDs=dicom.read_file(dicom_path_img)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns))
    ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
    ds = dicom.read_file(dicom_path_img)
    ArrayDicom[:, :] = ds.pixel_array
    ArrayDicom=ArrayDicom.astype('float')
    ArrayDicom=min_max_normalize(ArrayDicom)
    np.save(title, ArrayDicom)
    return(ArrayDicom)

def save_all_dicom_img(dicom_path_folder):
    subject=os.listdir(dicom_path_folder)
    for i in range(1,len(subject)+1):
        dicom_path_sub_folder_M=dicom_path_folder+subject[i-1]+'/M/'
        dicom_path_sub_folder_P=dicom_path_folder+subject[i-1]+'/P/'
        slices_M=os.listdir(dicom_path_sub_folder_M)
        slices_P=os.listdir(dicom_path_sub_folder_P)
        for j in range(1,len(slices_M)+1):
            dicom_path_img_M=dicom_path_sub_folder_M+slices_M[j-1]
            title='Lisa_complex_all/subject_'+str(i)+'_slice_'+str(j)+'_M'
            save_dicom(dicom_path_img_M,title)
            dicom_path_img_P=dicom_path_sub_folder_P+slices_P[j-1]
            title='Lisa_complex_all/subject_'+str(i)+'_slice_'+str(j)+'_P'
            save_dicom(dicom_path_img_P,title)
    return()
