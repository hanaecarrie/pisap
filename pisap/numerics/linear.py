##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
This module contains linears operators classes.
"""

# System import
import numpy
from scipy.signal import convolve2d

# Package import
import pisap.extensions.transform
from pisap.base.transform import WaveletTransformBase
from pisap.base.utils import extract_paches_from_2d_images
from sklearn.feature_extraction.image import reconstruct_from_patches_2d


class Identity():
    """ Identity operator class
    This is a dummy class that can be used in the optimisation classes
    """

    def __init__(self):
        self.l1norm = 1.0

    def op(self, data, **kwargs):
        """ Returns the input data unchanged

        Parameters
        ----------
        data : np.ndarray
            Input data array
        **kwargs
            Arbitrary keyword arguments
        Returns
        -------
        np.ndarray input data
        """
        return data

    def adj_op(self, data):
        """ Returns the input data unchanged

        Parameters
        ----------
        data : np.ndarray
            Input data array
        Returns
        -------
        np.ndarray input data
        """
        return data


class Wavelet(object):
    """ This class defines the wavelet transform operators.
    """
    def __init__(self, wavelet, nb_scale=4):
        """ Initialize the Wavelet class.

        Parameters
        ----------
        wavelet: str
            the wavelet to be used during the decomposition.
        nb_scales: int, default 4
            The number of scales in the decomposition.
        """
        self.nb_scale = nb_scale
        if wavelet not in WaveletTransformBase.REGISTRY:
            raise ValueError("Unknown tranformation '{0}'.".format(wavelet))
        self.transform = WaveletTransformBase.REGISTRY[wavelet](
            nb_scale=nb_scale, verbose=0)
            
    def set_coeff(self, coeff):
        self.transform.analysis_data = coeff

    def op(self, data):
        """ Operator.

        This method returns the input data convolved with the wavelet filter.

        Parameters
        ----------
        data: ndarray
            Input data array, a 2D image.

        Returns
        -------
        coeffs: ndarray
            The wavelet coefficients.
        """
        self.transform.data = data
        self.transform.analysis()
        return self.transform.analysis_data

    def adj_op(self, coeffs, dtype="array"):
        """ Adjoint operator.

        This method returns the reconsructed image.

        Parameters
        ----------
        coeffs: ndarray
            The wavelet coefficients.
        dtype: str, default 'array'
            if 'array' return the data as a ndarray, otherwise return a
            pisap.Image.

        Returns
        -------
        ndarray reconstructed data.
        """
        self.transform.analysis_data = coeffs
        image = self.transform.synthesis()
        if dtype == "array":
            return image.data
        return image

    def l2norm(self, data_shape):
        """ Compute the L2 norm.

        Parameters
        ----------
        data_shape: uplet
            the data shape.
        Returns
        -------
        norm: float
            the L2 norm.
        """
        # Create fake data.
        data_shape = numpy.asarray(data_shape)
        data_shape += data_shape % 2
        fake_data = numpy.zeros(data_shape)
        fake_data[zip(data_shape / 2)] = 1

        # Call mr_transform.
        data = self.op(fake_data)

        # Compute the L2 norm
        return numpy.linalg.norm(data)


class DictionaryLearningWavelet(object):
    """ This class defines the leaned wavelet transform operator based on a
    Dictionay Learning procedure.
    """

    def __init__(self, dictionary, atoms, img_shape, type_decomposition="prodscalar"):
        """ Initialize the Wavelet class.

        Parameters
        ----------
        dictonary: sklearn MiniBatchDictionaryLearning object
            containing the 'transform' method
        atoms: ndarray,
            ndarray of floats, 2d matrix dim nb_patches*nb_components,
            the learnt atoms
        img_shape= tuple of int, shape of the image
            (not necessarly a square image)
        type_decomposition: str, (default='prodscalar')
            should be ['prodscalar', 'convol'], specify how the decomposition is
            done.
        """
        #raise NotImplemented("plugg DL: WIP status for now")  # XXX
        self.dictionary = dictionary
        self.atoms = atoms
        self.type_decomposition = type_decomposition
        self.img_shape = img_shape
        self.coeff = [] #self.op(numpy.zeros(img_shape))
        self.coeffs_shape = []
    
    def set_coeff(self, coeff):
        self.coeff = coeff

    def op(self, image): #XXX works for square patches only!
        """ Operator.

        This method returns the representation of the input data in the
        learnt dictionary, that is to say the wavelet coefficients.

        Parameters
        ----------
        data: ndarray
            Input data array, a 2D image.

        Returns
        -------
        coeffs: ndarray of floats, 2d matrix dim nb_patches*nb_components,
                the wavelet coefficients.
        """
        coeffs = []
        patches_size = int(numpy.sqrt(self.atoms.shape[1])) #because square patches
        patches_shape = (patches_size,patches_size)
        patches = extract_paches_from_2d_images(image, patches_shape)
        coeffs = self.dictionary.transform(numpy.nan_to_num(patches))
        self.coeff = numpy.array(coeffs)
        return self.coeff

    def adj_op(self, coeffs, dtype="array"): #XXX works for square patches only!
        """ Adjoint operator.

        This method returns the reconsructed image from the wavelet coefficients. 

        Parameters
        ----------
        coeffs: ndarray of floats, 2d matrix dim nb_patches*nb_components,
                the wavelet coefficients.
                WARNING: CHECK THE COEFF DIMENSIONS!!!
                         the 'op' method returns the right shape of coeffs
        dtype: str, default 'array'
            if 'array' return the data as a ndarray, otherwise return a
            pisap.Image.

        Returns
        -------
        image_r: ndarray, the reconstructed data.
        """
        nb_patches = coeffs.shape[0]
        patches_size = int(numpy.sqrt(self.atoms.shape[1])) #because square patches
        image = numpy.dot(coeffs, self.atoms)
        image = image.reshape(nb_patches,patches_size, patches_size)
        image = reconstruct_from_patches_2d(image, self.img_shape)
        return image

    def l2norm(self, data_shape):
        """ Compute the L2 norm.

        Parameters
        ----------
        data_shape: uplet
            the data shape.
        Returns
        -------
        norm: float
            the L2 norm.
        """
        # Create fake data.
        data_shape = numpy.asarray(data_shape)
        data_shape += data_shape % 2
        fake_data = numpy.zeros(data_shape)
        fake_data[zip(data_shape / 2)] = 1

        # Call mr_transform.
        data = self.op(fake_data)

        # Compute the L2 norm
        return numpy.linalg.norm(data)


