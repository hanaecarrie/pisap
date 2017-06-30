# -*- coding: utf-8 -*-
##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
#
#:Author: Samuel Farrens <samuel.farrens@gmail.com>
#:Version: 1.1
#:Date: 04/01/2017
##########################################################################

"""
This module contains classes of different cost functions for optimization.
"""

# System import
from __future__ import print_function
import numpy as np
import os.path as osp
import pickle

# Package import
from pisap.plotting import plot_cost
from pisap.base.dictionary import DictionaryBase
from pisap.base.utils import generic_l1_norm

class costFunction():
    """ Cost function class

    This class implements the cost function for deconvolution

    Parameters
    ----------
    y : np.ndarray
        Input original data array
    grad : class
        Gradient operator class
    wavelet : class, optional
        Wavelet operator class ("sparse" mode only)
    weights : np.ndarray, optional
        Array of wavelet thresholding weights ("sparse" mode only)
    lambda_reg : float, optional
        Low-rank regularization parameter ("lowr" mode only)
    mode : str {'lowr', 'sparse'}, optional
        Deconvolution mode (default is "lowr")
    positivity : bool, optional
        Option to test positivity contraint (defult is "True")
    tolerance : float, optional
        Tolerance threshold for convergence (default is "1e-4")
    window : int, optional
        Iteration interval to test for convergence (default is "5")
    print_cost : bool, optional
        Option to print cost function value at each iteration (default is
        "True")
    output : str, optional
        Output file name for cost function plot

    """

    def __init__(self, y, grad, wavelet=None, weights=None,
                 lambda_reg=None, mode='lowr',
                 positivity=True, tolerance=1e-4, window=5, print_cost=True,
                 output=None):
        self.y = y
        self.grad = grad
        self.wavelet = wavelet
        self.lambda_reg = lambda_reg
        self.mode = mode
        self.positivity = positivity
        self.update_weights(weights)
        self.cost = -1
        self.regu = -1
        self.res = -1
        self.cost_list = []
        self.res_list = []
        self.regu_list = []
        self.x_list = []
        self.tolerance = tolerance
        self.print_cost = print_cost
        self.iteration = 0
        self.output = output
        self.window = window
        self.test_list = []

    def update_weights(self, weights):
        """ Update weights

        Update the values of the wavelet threshold weights ("sparse" mode only)

        Parameters
        ----------
        weights : np.ndarray
            Array of wavelet thresholding weights
        """
        self.weights = weights

    def l2norm(self, x):
        """ Calculate l2 norm

        This method returns the l2 norm error of the difference between the
        original data and the data obtained after optimisation

        Parameters
        ----------
        x : np.ndarray
            Deconvolved data array

        Returns
        -------
        float l2 norm value
        """
        #l2_norm = np.linalg.norm(self.y - self.grad.MX(x))
        l2_norm = np.var(self.y - self.grad.MX(x))
        #if self.print_cost:
        #    print(" - L2 NORM: ", l2_norm)
        return l2_norm

    def l1norm(self, x):
        """ Calculate l1 norm

        This method returns the l1 norm error of the weighted wavelet
        coefficients

        Parameters
        ----------
        x : np.ndarray
            Deconvolved data array

        Returns
        -------
        float l1 norm value

        """
        if isinstance(x, DictionaryBase):
            y = x
            l1_norm = np.sum(y.absolute.to_cube()).real
        elif isinstance(x, np.ndarray):
            y = x.flatten()
            l1_norm = np.sum(np.abs(y))
        else:
            raise TypeError("l1norm can only be compute on DictionaryBase or np.ndarray")
        if self.print_cost:
            print("  - l1 norm: ", l1_norm)
        return l1_norm

    def nucnorm(self, x):
        """ Calculate nuclear norm

        This method returns the nuclear norm error of the deconvolved data in
        matrix form

        Parameters
        ----------
        x : np.ndarray
            Deconvolved data array

        Returns
        -------
        float nuclear norm value

        """
        x_prime = cube2matrix(x)
        nuc_norm = nuclear_norm(x_prime)
        if self.print_cost:
            print(" - nuclear norm: ", nuc_norm)
        return nuc_norm

    def check_cost(self, x):
        """ Check cost function

        This method tests the cost function for convergence in the specified
        interval of iterations

        Parameters
        ----------
        x : np.ndarray
            Deconvolved data array

        Returns
        -------
        bool result of the convergence test

        """
        if self.iteration % (2 * self.window):
            self.x_list.append(x)
            self.test_list.append(self.cost)
            return False

        else:
            self.x_list.append(x)
            self.test_list.append(self.cost)
            #x1 = np.average(self.x_list[:self.window], axis=0)
            #x2 = np.average(self.x_list[self.window:], axis=0)
            t1 = np.average(self.test_list[:self.window], axis=0)
            t2 = np.average(self.test_list[self.window:], axis=0)
            self.x_list = []
            self.test_list = []

            test = (np.linalg.norm(t1 - t2) / np.linalg.norm(t1))

            if self.print_cost:
                print("  - convergence test: ", test)

            return test <= self.tolerance

    def check_residual(self, x):
        """ Check residual

        This method calculates the residual between the deconvolution and the
        observed data

        Parameters
        ----------
        x : np.ndarray
            Deconvolved data array
        """
        #self.res = np.std(self.y - self.grad.MX(x)) # / np.linalg.norm(self.y)
        if self.print_cost:
            #print(" - STD RESIDUAL: ", self.res)
            print("  - lasso cost: ", self.cost)
            print("  - res cost: ", self.res)
            print("  - regu cost: ", self.regu)

    def get_cost(self, x):
        """ Get cost function

        This method calculates the full cost function and checks the result for
        convergence

        Parameters
        ----------
        x : np.ndarray
            Deconvolved data array

        Returns
        -------
        bool result of the convergence test
        """
        if self.print_cost:
            print(" - ITERATION: ", self.iteration + 1)

        self.iteration += 1
        self.cost_old = self.cost

        if self.positivity:
            print("  - min(X): ", np.min(x))

        if self.mode == 'all':
            self.cost = (0.5 * self.l2norm(x) ** 2 + self.l1norm(x) +
                         self.nucnorm(x))

        elif self.mode == 'lasso':
            self.res = 0.5 * np.linalg.norm(self.grad.MX(x) - self.y)**2
            self.regu = generic_l1_norm(self.wavelet.op(x))
            self.cost = self.res + self.lambda_reg * self.regu

        elif self.mode == 'lowr':
            self.cost = (0.5 * self.l2norm(x) ** 2 + self.lambda_reg *
                         self.nucnorm(x))

        elif self.mode == 'grad':
            self.cost = 0.5 * self.l2norm(x) ** 2

        self.cost_list.append(self.cost)
        self.res_list.append(self.res)
        self.regu_list.append(self.regu)

        self.check_residual(x)

        return self.check_cost(x)

    def pickle_cost_list(self, filename=None):
        """ Save the parameter of the optimisation and the cost list value in a
            pickle file.
        """
        if filename is None:
            filename = osp.splitext(self.output)[0] + ".pkl"
        to_dump = {'nb_iter': self.iteration,
                   'mu': self.lambda_reg,
                   'cost_list': np.array(self.cost_list),
                   'regu_list': np.array(self.regu_list),
                   'res_list': np.array(self.res_list),
                   }
        with open(filename, "wb") as pfile:
            pickle.dump(to_dump, pfile)

    def plot_cost(self):
        """ Plot cost function.

        This method plots the cost function as function of iteration number.
        """
        plot_cost(self.cost_list, self.output)


def cube2matrix(data_cube):
    """ This method transforms a 3D cube to a 2D matrix

    Parameters
    ----------
    data_cube : np.ndarray
        Input data cube, 3D array
    Returns
    -------
    np.ndarray 2D matrix
    """
    return data_cube.reshape([data_cube.shape[0]] +
        [np.prod(data_cube.shape[1:])]).T


def nuclear_norm(data):
    """ Function that computes the nuclear norm of the input data.

    Parameters
    ----------
    data : np.ndarray
        Input data
    Returns
    -------
    Nuclear norm
    """

    # Get SVD of the data.
    u, s, v = np.linalg.svd(data)

    # Return nuclear norm.
    return np.sum(s)

