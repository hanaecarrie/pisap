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
import numpy as np
from scipy.signal import convolve2d
from sklearn.decomposition import sparse_encode
from sklearn.feature_extraction.image import extract_patches_2d, \
                                             reconstruct_from_patches_2d

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
        data_shape = np.asarray(data_shape)
        data_shape += data_shape % 2
        fake_data = np.zeros(data_shape)
        fake_data[zip(data_shape / 2)] = 1

        # Call mr_transform.
        data = self.op(fake_data)

        # Compute the L2 norm
        return np.linalg.norm(data)


class DictionaryLearningWavelet(object):
    """ This class defines the leaned wavelet transform operator based on a
    Dictionay Learning procedure.
    """

    def __init__(self, atoms, image_size, alpha_transform=0.1,
                 max_iter_transform=50, n_jobs_transform=1,
                 verbose_transform=False):
        """ Initialize the Wavelet class.

        Parameters
        ----------
        atoms: ndarray,
            ndarray of three dimensions, defining the 2D patchs or images that
            composed the dictionary.
        image_size: tuple of int,
            size of the considerated image.
        alpha_transform: float, (optional, default 1.0e-2)
            the sparcity regulartization parameter for the sparse_encode in op.
        max_iter_transform: int, (optional, default 100)
            maximum number of iteration for the lasso minimization in the op.
        n_jobs_transform: int, (optional, default -1)
            Number of parallel jobs to run.
        verbose_transform: int, (optional, default False)
            level of verbose, contaminated all the subfunctions.
        """
        d1, d2, d3 = atoms.shape
        self.size_patches = (d1, d2)
        self.dictionary = atoms.reshape(d3, d1*d2)
        self.image_size = image_size
        self.alpha_transform = alpha_transform
        self.max_iter_transform = max_iter_transform
        self.verbose_transform = verbose_transform
        self.n_jobs_transform = n_jobs_transform

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
        if np.any(np.iscomplex(data)):
            r_coef = self._op_real_data(data.real)
            i_coef = self._op_real_data(data.imag)
            return r_coef + 1.j * i_coef
        else:
            return self._op_real_data(data.astype('float64'))

    def _op_real_data(self, data):
        """ Operator for real data, private method.

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
        patches = extract_patches_2d(data, self.size_patches)
        d1, d2, d3 = patches.shape
        patches_reshaped = patches.reshape(d1, d2*d3)
        coef = sparse_encode(
                  patches_reshaped, self.dictionary,
                  n_jobs=self.n_jobs_transform, alpha=self.alpha_transform,
                  max_iter=self.max_iter_transform,
                  verbose=self.verbose_transform)
        return coef

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
        patches = np.dot(coefs, self.dictionary)
        d1, d2 = patches.shape
        patches = patches.reshape(d1, int(np.sqrt(d2)), int(np.sqrt(d2)))
        if np.any(np.iscomplex(patches)):
            r_patches = patches.real
            i_patches = patches.imag
            r_img = reconstruct_from_patches_2d(r_patches, self.image_size)
            i_img = reconstruct_from_patches_2d(i_patches, self.image_size)
            img = r_img + 1.j * i_img
        else:
            img = reconstruct_from_patches_2d(patches.astype('float64'), self.image_size)
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
        data_shape = np.asarray(data_shape)
        data_shape += data_shape % 2
        fake_data = np.zeros(data_shape)
        fake_data[zip(data_shape / 2)] = 1

        # Call mr_transform.
        data = self.op(fake_data)

        # Compute the L2 norm
        return np.linalg.norm(data)


