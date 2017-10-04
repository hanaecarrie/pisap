# -*- coding: utf-8 -*-
##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
"""
This module contains classses for defining algorithm operators and gradients.
"""
# System import
import copy
import numpy as np
import scipy.fftpack as pfft

import pisap


class GradBase(object):
    """ Basic gradient class

    This class defines the basic methods that will be inherited by specific
    gradient classes
    """
    def get_initial_x(self):
        """ Set initial value of x.

        This method sets the initial value of x to an arrray of random values
        """
        raise NotImplementedError("'GradBase' is an abstract class: " \
                                    +   "it should not be instanciated")

    def MX(self, x):
        """ MX

        This method calculates the action of the matrix M on the data X, in
        this case fourier transform of the the input data

        Parameters
        ----------
        x : np.ndarray
            Input data array, an array of recovered 2D images

        Returns
        -------
        np.ndarray result
        """
        raise NotImplementedError("'GradBase' is an abstract class: " \
                                    +   "it should not be instanciated")

    def MtX(self, x):
        """ MtX

        This method calculates the action of the transpose of the matrix M on
        the data X, in this case inverse fourier transform of the input data

        Parameters
        ----------
        x : np.ndarray
            Input data array, an array of recovered 2D images

        Returns
        -------
        np.ndarray result
        """
        raise NotImplementedError("'GradBase' is an abstract class: " \
                                    +   "it should not be instanciated")

    def get_spec_rad(self, tolerance=1e-4, max_iter=20, coef_mul=1.1):
        """ Get spectral radius.

        This method calculates the spectral radius.

        Parameters
        ----------
        tolerance : float (optional, default 1e-8)
            Tolerance threshold for convergence.
        max_iter : int (optional, default 150)
            Maximum number of iterations.
        verbose: int (optional, default 0)
            The verbosity level.
        """
        # Set (or reset) values of x.
        x_old = self.get_initial_x()

        # Iterate until the L2 norm of x converges.
        for i in xrange(max_iter):
            x_new = self.MtMX(x_old) / np.linalg.norm(x_old)
            if(np.abs(np.linalg.norm(x_new) - np.linalg.norm(x_old)) < tolerance):
                break
            x_old = copy.deepcopy(x_new)
        self.spec_rad = coef_mul * np.linalg.norm(x_new)
        self.inv_spec_rad = 1.0 / self.spec_rad

    def MtMX(self, x):
        """ M^T M X

        This method calculates the action of the transpose of the matrix M on
        the action of the matrix M on the data X

        Parameters
        ----------
        x : np.ndarray
            Input data array

        Returns
        -------
        np.ndarray result

        Notes
        -----
        Calculates  M^T (MX)
        """
        return self.MtX(self.MX(x))


    def get_grad(self, x):
        """ Get the gradient step

        This method calculates the gradient step from the input data

        Parameters
        ----------
        x : np.ndarray
            Input data array

        Returns
        -------
        np.ndarray gradient value

        Notes
        -----

        Calculates M^T (MX - Y)
        """
        self.grad = self.MtX(self.MX(x) - self.y)


class Grad2DAnalysis(GradBase):
    """ Analysis 2D gradient class

    This class defines the operators for a 2D array

    Parameters
    ----------
    data : np.ndarray
        Input data array, an array of 2D observed images (i.e. with noise)
    ft_cls :  Fourier operator derive from base class 'FourierBase'
        Fourier class, for computing Fourier transform (NFFT or FFT)
    """
    def __init__(self, data, ft_cls):
        """ Initilize the Grad2DSynthese class.
        """
        self.y = data
        if isinstance(ft_cls, dict):
            if len(ft_cls) > 1:
                raise ValueError("ft_cls in Grad2DAnalyse should ether be a 1"
                                 " 'key dict' or a 'fourier op class'")
            self.ft_cls = ft_cls.keys()[0](**ft_cls.values()[0])
        else:
            self.ft_cls = ft_cls
        self.get_spec_rad()

    def get_initial_x(self):
        """ Set initial value of x.

        This method sets the initial value of x to an arrray of random values
        """
        shape = self.ft_cls.img_shape
        return np.random.random(shape).astype(np.complex)

    def MX(self, x):
        """ MX

        This method calculates the action of the matrix M on the data X, in
        this case fourier transform of the the input data

        Parameters
        ----------
        x : np.ndarray
            Input data array, an array of recovered 2D images

        Returns
        -------
        np.ndarray result
        """
        return self.ft_cls.op(x)

    def MtX(self, x):
        """ MtX

        This method calculates the action of the transpose of the matrix M on
        the data X, in this case inverse fourier transform of the input data

        Parameters
        ----------
        x : np.ndarray
            Input data array, an array of recovered 2D images

        Returns
        -------
        np.ndarray result
        """
        return self.ft_cls.adj_op(x)


class Grad2DSynthesis(GradBase):
    """ Synthesis 2D gradient class

    This class defines the grad operators for |M*F*invL*alpha - data|**2.

    Parameters
    ----------
    data : np.ndarray
        Input data array, an array of 2D observed images (i.e. with noise)
    ft_cls :  Fourier operator derive from base class 'FourierBase'
        Fourier class, for computing Fourier transform (NFFT or FFT)
    linear_cls: class
        a linear operator class.
    """
    def __init__(self, data, ft_cls, linear_cls):
        """ Initilize the Grad2DAnalyse class.
        """
        self.y = data
        if isinstance(ft_cls, dict):
            if len(ft_cls) > 1:
                raise valueerror("ft_cls in grad2danalyse should ether be a"
                                 " 'key dict' or a 'fourier op class'")
            self.ft_cls = ft_cls.keys()[0](**ft_cls.values()[0])
        else:
            self.ft_cls = ft_cls
        self.linear_cls = linear_cls
        self.get_spec_rad()

    def get_initial_x(self):
        """ Set initial value of x.

        This method sets the initial value of x to an arrray of random values
        """
        shape = self.ft_cls.img_shape
        fake_data = np.zeros(shape).astype(np.complex)
        wt_coef = self.linear_cls.op(fake_data)
        return np.random.random(wt_coef.shape).astype(np.complex)

    def MX(self, alpha):
        """ MX

        This method calculates the action of the matrix M on the data X, in
        this case fourier transform of the the input data

        Parameters
        ----------
        alpha : DictionaryBase
            Input analysis decomposition

        Returns
        -------
        np.ndarray result recovered 2D kspace
        """
        return self.ft_cls.op(self.linear_cls.adj_op(alpha))

    def MtX(self, x):
        """ MtX

        This method calculates the action of the transpose of the matrix M on
        the data X, in this case inverse fourier transform of the input data in
        the frequency domain.

        Parameters
        ----------
        x : np.ndarray
            Input data array, an array of recovered 2D kspace

        Returns
        -------
        DictionaryBase result
        """
        return self.linear_cls.op(self.ft_cls.adj_op(x))
