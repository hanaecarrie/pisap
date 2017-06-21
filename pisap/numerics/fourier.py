# -*- coding: utf-8 -*-
##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
"""
This module contains classses for defining algorithm Fourier transform operators
for both case equispaced and non-equasipaced sampling
"""

import numpy as np
import scipy.fftpack as pfft
import pynfft
from pisap.base.utils import convert_locations_to_mask

class FourierBase(object):
    """ Basic gradient class

    This class defines the basic methods that will be inherited by specific
    gradient classes
    """

    def op(self, img):
        """ This method calculates Fourier transform of the 2-D argument img

        Parameters
        ----------
        img : np.ndarray of dim 2
            Input image as array

        Returns
        -------
        np.ndarray Fourier transform of the image
        """
        raise NotImplementedError("'FourierBase' is an abstract class: " \
                                    +   "it should not be instanciated")

    def adj_op(self, x):
        """This method calculates inverse Fourier transform of real or complex
        sequence

        Parameters
        ----------
        x : np.ndarray
            Input data array, an array of recovered 2D images

        Returns
        -------
        np.ndarray inverse two-dimensionnal discrete Fourier transform of
        arbitrary type sequence x
        """
        raise NotImplementedError("'FourierBase' is an abstract class: " \
                                    +   "it should not be instanciated")


class FFT(FourierBase):
    """ Standard 2D Fast Fourrier Transform class

    This class defines the operators for a 2D array

    Attributes
    ----------
    samples_locations :  np.ndarray
        The subsampling mask in the Fourier domain.
    img_size :
    """

    def __init__(self, samples_locations, img_size):
        """ Initilize the Grad2DSynthese class.
        """
        self.samples_locations = samples_locations
        self.img_size = img_size
        self._mask = convert_locations_to_mask(self.samples_locations, img_size)

    def op(self, img):
        """ This method calculates Masked Fourier transform of the 2-D argument
        img

        Parameters
        ----------
        img : np.ndarray of dim 2
            Input image as array

        Returns
        -------
        np.ndarray Fourier transform of the image
        """
        return self._mask * pfft.fft2(img)

    def adj_op(self,x):
        """ This method calculates inverse Fourier transform of real or complex
        masked sequence

        Parameters
        ----------
        x : np.ndarray of dim 2
            Input image as array

        Returns
        -------
        np.ndarray inverse two-dimensionnal discrete Fourier transform of
        arbitrary type sequence x
        """
        return pfft.ifft2(self._mask * x)


class NFFT(FourierBase):
    """ Standard 2D Fast Fourrier Transform class

    This class defines the operators for a 2D array

    Attributes
    ----------
    samples_locations :  np.ndarray
        The subsampling mask in the Fourier domain.
    img_size : int
        Image size
    """

    def __init__(self, samples_locations, img_size):
        """ Initilize the Grad2DSynthese class.
        """
        self.plan = pynfft.NFFT(N = (img_size, img_size), M=len(samples_locations))
        self.img_size = img_size
        self.samples_locations = samples_locations
        self.plan.x = self.samples_locations
        self.plan.precompute()

    def op(self, img):
        """ This method calculates Masked Fourier transform of the 2-D argument
        img

        Parameters
        ----------
        img : np.ndarray of dim 2
            Input image as array

        Returns
        -------
        np.ndarray Fourier transform of the image
        """
        self.plan.f_hat = img
        return self.plan.trafo()

    def adj_op(self,x):
        """ This method calculates inverse Fourier transform of real or complex
        masked sequence

        Parameters
        ----------
        x : np.ndarray of dim 2
            Input image as array

        Returns
        -------
        np.ndarray inverse two-dimensionnal discrete Fourier transform of
        arbitrary type sequence x
        """
        self.plan.f = x
        return (1.0/self.plan.M) * self.plan.adjoint()
