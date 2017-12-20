""" Patches.
"""

# Sys import
import numpy as np

# Third party import
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

# Package import
from pisap.base.utils import min_max_normalize
from pisap.base.utils import extract_patches_from_2d_images


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
            if option == 'abs':
                image = np.abs(img).astype('float')
                patches = extract_patches_from_2d_images(min_max_normalize(image),
                                                         patch_shape)
            elif option == 'real':
                image = np.real(img)
                patches = extract_patches_from_2d_images(min_max_normalize(image),
                                                         patch_shape)
            elif option == 'imag':
                image = np.imag(img)
                patches = extract_patches_from_2d_images(min_max_normalize(image),
                                                         patch_shape)
            else:
                patches_r = extract_patches_from_2d_images(
                                        min_max_normalize(np.real(img)),
                                        patch_shape)
                patches_i = extract_patches_from_2d_images(
                                        min_max_normalize(np.imag(img)),
                                        patch_shape)
                patches = patches_r + 1j * patches_i

            flat_patches_sub.append(patches)
        yield flat_patches_sub


def reconstruct_2d_images_from_flat_patched_images(flat_patches_list,
                                                   patch_size, dico,
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
    # patch_size = int(np.sqrt(flat_patches_list[0].shape[1]))
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
