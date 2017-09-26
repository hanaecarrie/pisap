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
from sklearn.decomposition import sparse_encode
from sklearn.feature_extraction.image import extract_patches_2d

# Package import
import pisap.extensions.transform
from pisap.base.transform import WaveletTransformBase


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


class DictionnaryLearningWavelet(object):
    """ This class defines the leaned wavelet transform operator based on a
    Dictionnay Learning procedure.
    """

    def __init__(self, atoms, img_size):
        """ Initialize the Wavelet class.

        Parameters
        ----------
        atoms: ndarray,
            ndarray of three dimensions, defining the 2D patchs or images that
            composed the dictionnary.
        img_size: tuple of int,
            the shape of the considered images.
        """
        self.img_size = img_size
        d1, d2, d3 = atoms.shape
        self.dictionary = atoms.reshape(d3, d1*d2)

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
        patches = extract_patches_2d(data, self.img_size)
        d1, d2, d3 = patches.shape
        data = patches.reshape(d3, d1*d2)
        return sparse_encode(data, self.dictionary)

    def adj_op(self, coefs, dtype="array"):
        """ Adjoint operator.

        This method returns the reconsructed image.

        Parameters
        ----------
        coefs: ndarray
            The wavelet coefficients.
        dtype: str, default 'array'
            if 'array' return the data as a ndarray, otherwise return a
            pisap.Image.

        Returns
        -------
        ndarray reconstructed data.
        """
        d1, d2 = coefs.shape
        patches = coefs.reshape(np.sqrt(d2), np.sqrt(d2), d1)
        img = reconstruct_from_patches_2d(patches, self.image_size)
        if dtype == "array":
            return img
        return Image(data=img)

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


