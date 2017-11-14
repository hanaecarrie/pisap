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


def compare_dictionaries_SparseCode(CV_id, threshold=0):
    """ Load the previousely learnt dictionaries
        Compute the sparse code for the validation_set
        Reconstruct the image from the sparse code

    Parameters:
    -----
        CV_id: str, name of the crossvalidation folder in
            '/home/hc253658/STAGE/Crossvalidations/'
        threshold (default =0): thresholding level of the sparse coefficents
    Return:
    -------
        str, massage to indicate that the algorithm successfully ended
    """
    path = '/home/hc253658/STAGE/Crossvalidations/'+CV_id
    nb_cv_iter=len(os.listdir(path+'/'))

    for i in range(1,nb_cv_iter+1):
        dicos_list = os.listdir(path+'/iter_'+str(i)+'/dicos/')
        validation_set=np.load(path+'/iter_'+str(i)+'/refs_val/refs_val.npy')
        validation_set=validation_set.tolist()
        for val in validation_set:
            val=np.array(val)
        img_shape = np.array(validation_set[0]).shape
        index_val=np.load(path+'/iter_'+str(i)+'/refs_val/index_val.npy')

        for dico in dicos_list:
            dico_folder=path+'/iter_'+str(i)+'/dicos/'+dico
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
