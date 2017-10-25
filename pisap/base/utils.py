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
    img=np.nan_to_num(img)
    min_img = img.min()
    max_img = img.max()
    img=(img - min_img) / (max_img - min_img)
    img=np.nan_to_num(img)
    return img


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
    samples_locations = np.round(samples_locations)
    samples_locations = samples_locations.astype('int')
    mask = np.zeros(img_shape)
    mask[samples_locations[:,0], samples_locations[:,1]] = 1
    return mask


def extract_patches_from_2d_images(img, patch_shape): # XXX the patches need to be square for the reshape
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
    #patches -= np.mean(patches, axis=0)
    #patches /= np.std(patches, axis=0)
    return patches
    
def subsampling_op(kspace, mask):
    """ Return the samples from the Cartesian kspace after undersampling
    with the mask

    Parameters:
    -----------
        kspace: np.ndarray of complex, 2d matrix
        mask: np.ndarray of int, {0,1} 2d matrix
    Returns:
    -------
        samples=np.array of complex, column of size the number of samples
        of the mask 
    """
    row, col = np.where(mask==1)
    samples = kspace[row,col]
    return samples
    
def subsampling_adj_op(samples, mask):
    """ Return the kspace corresponding to the samples and the 2d mask 

    Parameters:
    -----------
        samples: np.array of complex,  column of size the number of samples
        of the mask 
        mask: np.ndarray of int, {0,1} 2d matrix
    Returns:
    -------
        kspace=np.ndarray of complex, 2d matrix, the undersampled kspace
    """
    row, col = np.where(mask==1)
    kspace = np.zeros(mask.shape).astype('complex128')
    kspace[row,col] = samples
    return kspace
    
def crop_sampling_scheme(sampling_scheme, img_shape):
    """Crop the sampling scheme from a input sampling scheme
    with an even wigth and hight
    ---------
    Inputs:
    sampling_scheme -- np.ndarray of {0,1}, 2d matrix, the sampling scheme
    ---------
    Outputs:
    sampling_scheme -- np.ndarray of {0,1}, 2d matrix, the cropped
                       sampling scheme
    """
    h = img_shape[1]
    w = img_shape[0]
    ss_size=sampling_scheme.shape
    sampling_scheme = sampling_scheme[
                            ss_size[0]/2-int((w+1)/2):ss_size[0]/2+int((w)/2),
                            ss_size[1]/2-int((h+1)/2):ss_size[1]/2+int((h)/2)]
    return sampling_scheme