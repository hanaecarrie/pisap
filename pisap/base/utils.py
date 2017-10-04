""" Module that declare usefull metrics tools.
"""
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d

def min_max_normalize(img):
    """ Center and normalize the given array.
    Parameters:
    ----------
    img: np.ndarray
    """
    min_img = img.min()
    max_img = img.max()
    return (img - min_img) / (max_img - min_img)


def convert_mask_to_locations(mask):
    """ Return the converted Cartesian mask as sampling locations.

    Parameters:
    -----------
        mask: np.ndarray, {0,1} 2D matrix, not necessarly a square matrix
    Returns:
    -------
        samples_locations: np.ndarray,
    """
    row, col = np.where(mask==1)
    row = row.astype('float') / mask.shape[0] - 0.5
    col = col.astype('float') / mask.shape[1] - 0.5
    return np.c_[row, col]


def convert_locations_to_mask(samples_locations, img_shape):
    """ Return the converted the sampling locations as Cartesian mask.

    Parameters:
    -----------
        samples_locations: np.ndarray,
        img_shape: tuple of int, shape of the desired mask, not necessarly
        a square matrix
    Returns:
    -------
        mask: np.ndarray, {0,1} 2D matrix
    """
    samples_locations = samples_locations.astype('float')
    samples_locations += 0.5
    samples_locations[:,0] *= img_shape[0]
    samples_locations[:,1] *= img_shape[1]
    samples_locations = samples_locations.astype('int')
    mask = np.zeros(img_shape)
    mask[samples_locations[:,0], samples_locations[:,1]] = 1
    return mask


def extract_paches_from_2d_images(img, patch_shape): # XXX the patches need to be square for the reshape
    """ Return the flattened patches from the 2d image.

    Parameters:
    -----------
        img: np.ndarray of floats, the input 2d image
        patch_shape: tuple of int, shape of the patches 
    Returns:
    -------
        patches: np.ndarray of floats, a 2d matrix with
        dim nb_patches*(patch.shape[0]*patch_shape[1])
    """
    patches  = extract_patches_2d(img, patch_shape)
    patches = patches.reshape(patches.shape[0], -1)
    patches -= np.mean(patches, axis=0)
    patches /= np.std(patches, axis=0)
    return patches
