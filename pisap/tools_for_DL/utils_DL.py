from pisap.tools_for_DL.dictionary import Dictionary
from pisap.tools_for_DL.images import LisaImages_forCV
import numpy as np
import os
import pickle

def generate_dictionary(images, refs_train, alpha, nb_atoms, patch_size):
    """Learn the dictionary from the complex images
    -----------
    Inputs:
        images -- instance of class CamCanImages, the preprocessed
            input images
        ref_train --
        n_components -- int, number of atoms
        WARNING: CHOOSE A SQUARE NUMBER, IF NOT THE PLOTTING METHOD WILL BUG
        regulation term (default=1)
        n_iter -- int, number of iterations (default=100)
        alpha -- alpha (default=1)
    -----------
    Outputs:
        dictionary_real, dictionary_imag
    """
    images.patch_shape = (patch_size, patch_size)
    print(len(refs_train))
    images.extract_patches_and_flatten_imgs_trainingset(refs_train)
    dictionary_real = Dictionary(nb_atoms, alpha, 100)
    dictionary_imag = Dictionary(nb_atoms, alpha, 100)
    dictionary_real.learning_atoms(images,'real')
    dictionary_imag.learning_atoms(images,'imaginary')

    return dictionary_real, dictionary_imag

def generate_dictionary_and_save_it(images, refs_train, alpha, nb_atoms,
                                patch_size, saving_path):

    dico_folder = saving_path+'/patch_'+str(patch_size)+'_atoms_'+str(nb_atoms)\
                    +'_alpha_'+str(alpha)
    os.makedirs(dico_folder)

    dictionary_real, dictionary_imag = generate_dictionary(images, refs_train,\
                                alpha, nb_atoms, patch_size)
    f=open(dico_folder+'/dico_real.p','w')
    pickle.dump(dictionary_real,f)
    f.close()
    f=open(dico_folder+'/dico_imag.p','w')
    pickle.dump(dictionary_imag,f)
    f.close()



# def shuffle_image_set():
