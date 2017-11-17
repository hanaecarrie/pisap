""" Module that declare usefull metrics tools.
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from sklearn.decomposition import MiniBatchDictionaryLearning
import time


def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

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


def extract_patches_from_2d_images(img, patch_shape): #XXX the patches need to be square
    """ Return the flattened patches from the 2d image.

    Parameters:
    -----------
        img: np.ndarray of floats, the input 2d image
        patch_shape: tuple of int, shape of the patches
    Returns:
    -------
        patches: np.ndarray of floats, a 2d matrix with
        -        dim nb_patches*(patch.shape[0]*patch_shape[1])
    """
    patches  = extract_patches_2d(img, patch_shape)
    patches = patches.reshape(patches.shape[0], -1)
    #patches -= np.mean(patches, axis=0)
    #patches /= np.std(patches, axis=0)
    return patches


def generate_flat_patches(images_training_set,patch_size, option='real'):
    """Learn the dictionary from the real/imaginary part or the module/phase of images
    from the training set
    -----------
    Inputs:
        images_training_set -- list of 2d matrix of complex values
        patch_size -- int, width of square patches
        option -- 'real' (default), 'imag' real/imaginary part or module/phase,
            'complex'
    -----------
    Outputs:
        flat_patches -- list of 1d array of flat patches (floats)
    """
    patch_shape = (patch_size, patch_size)
    flat_patches=[]
    print 'flat_patches starting...'
    for i in range(len(images_training_set)):
        if i%10 == 0:
            print(i)
        if option=='real':
            img=np.real(images_training_set[i])
        if option=='imag':
            img=np.imag(images_training_set[i])
        if option=='complex':
            img=images_training_set[i]
        patches_i=extract_patches_from_2d_images(min_max_normalize(img),patch_shape)
        flat_patches.append(patches_i)
    print 'flat_patches ended...'
    return(flat_patches)

def reconstruct_2d_images_from_flat_patched_images(flat_patches_list, dico, img_shape, threshold=1): #XXX the patches need to be square
    """ Return the list of 2d reconstructed images from sparse code
        (flat patches and dico)

    Parameters:
    -----------
        flat_patches_list: list of 2d np.ndarray of size nb_patches*len_patches
        dico: sklearn MiniBatchDictionaryLearning object
        img_shape: tuple of int, image shape
        threshold (default =1): thresholding level of the sparse coefficents
    Returns:
    -------
        reconstructed_images: list of 2d np.ndarray of size w*h
    """
    reconstructed_images=[]
    patch_size=int(np.sqrt(flat_patches_list[0].shape[1]))
    for i in range(len(flat_patches_list)):
        loadings = dico.transform(flat_patches_list[i])
        norm_loadings = min_max_normalize(np.abs(loadings))
        if threshold > 0:
            loadings[norm_loadings < threshold] = 0
        recons = np.dot(loadings, dico.components_)
        recons = recons.reshape((recons.shape[0],patch_size,patch_size))
        recons = reconstruct_from_patches_2d(recons,img_shape)
        reconstructed_images.append(recons)
    return reconstructed_images

def generate_dico(flat_patches, nb_atoms, alpha, n_iter,
                  fit_algorithm='lars',transform_algorithm='lars',n_jobs=-1):
    """Learn the dictionary from the real/imaginary part or the module/phase of images
    from the training set
    -----------
    Inputs:
        flat_patches -- list of 1d array of flat patches (floats)
        nb_atoms -- int, number of components of the dictionary
        alpha -- float, regulation term (default=1)
        n_iter -- int, number of iterations (default=100)
    -----------
    Outputs:
        dico -- object
    """
    dico=MiniBatchDictionaryLearning(n_components=nb_atoms, alpha=alpha,
                                     n_iter=n_iter, fit_algorithm=fit_algorithm,
                                     transform_algorithm=transform_algorithm,
                                     n_jobs=n_jobs)
    buffer = []
    index = 0
    print 'learning_atoms starting...'
    t_start = time.clock()
    for _ in range(6):
        for i in range(len(flat_patches)):
            index += 1
            patches=flat_patches[i]
            buffer.append(patches)
            if index % 10 == 0:
                print(str(index)+'/'+str(6*len(flat_patches)))
                patches = np.concatenate(buffer, axis=0)
                dico.fit(patches)
                buffer = []
    print 'learning_atoms ended!'
    t_end = time.clock()
    elapsed_time=timer(t_start,t_end)
    print 'Dictionary learnt in ', elapsed_time, 'seconds'
    return(dico)
    #RQ: atoms=dico.components_

def plot_dico(dico, patch_size,title='Dictionary atoms'):
    """Plot the learnt atoms
    -----------
    Inputs:
     dico -- dictionary object
     title -- string, (default='Dictionary atoms'), the .npy file title
    -----------
    Outputs:
     nothing
    """
    patch_shape=(patch_size,patch_size)
    n_components=dico.components_.shape[0]
    plt.figure(figsize=(4.2, 4))
    for i, patch in enumerate(dico.components_):
        plt.subplot(np.int(np.sqrt(n_components)),
                 np.int(np.sqrt(n_components)), i+1)
        plt.imshow(patch.reshape(patch_shape), cmap=plt.cm.gray,
                interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    plt.show()

def plot_img(image):
    """Plot a given image
    -----------
    Inputs:
     image -- 2d image of floats
    -----------
    Outputs:
     nothing
    """
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.colorbar()
    plt.show()

def _normalize_localisations(loc):
    """ Normalize localisation to [-0.5, 0.5[.
    """
    Kmax = loc.max()
    Kmin = loc.min()
    if Kmax < np.abs(Kmin):
        return loc / (2 * np.abs(Kmin) )
    else:
        loc[loc == Kmax] = -Kmax
        return loc / (2 * np.abs(Kmax) )

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


def save_object(obj, filename):
    """Save object into pickle format
    ---------
    Inputs:
    obj -- variable, name of the object in the current workspace
    filename -- string, filename to save the object
    ---------
    Outputs:
    nothing
    """
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    """Load object saved into pickle format
    ---------
    Inputs:
    filename -- string, path where the pickle object is saved
    ---------
    Outputs:
    nothing
    """
    f = open(filename, 'rb')
    loaded_obj = pickle.load(f)
    f.close()
    return(loaded_obj)
