# -*- coding: utf-8 -*-
##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
"""
This module contains classes of different cost/metric functions for optimization.
"""
import numpy as np

from pisap.base.utils import generic_l1_norm, generic_l2_norm
from meri.metric import compute_ssim, compute_snr, compute_psnr, compute_nrmse


class LASSOAnalysis:
    """ Declare the LASSO cost function w.r.t a specific cost function.
    """

    def __init__(self, y, grad, linear_op, lbda):
        """ Init.

        Parameters:
        -----------
        y: np.ndarray,
            the acquired kspace.

        grad: PISAP gradient class,
            the gradient operator that define the measurement matrix.

        linear_op: PISAP wavelet operator,
            the wavelet operator where regularization is done.

        lbda: float,
            the regularization parameter.
        """
        self.y = y
        self.grad = grad
        self.linear_op = linear_op
        self.lbda = lbda

    def __call__(self, x):
        """ Returnt the LASSO cost.
        """
        res =  0.5 * np.linalg.norm(self.grad.MX(x) - self.y)**2
        regu = generic_l1_norm(self.linear_op.op(x))
        return res + self.lbda * regu


class LASSOSynthesis:
    """ Declare the LASSO cost function w.r.t a specific cost function.
    """

    def __init__(self, y, grad, lbda):
        """ Init.

        Parameters:
        -----------
        y: np.ndarray,
            the acquired kspace.

        grad: PISAP gradient class,
            the gradient operator that define the measurement matrix.

        lbda: float,
            the regularization parameter.
        """
        self.y = y
        self.grad = grad
        self.lbda = lbda

    def __call__(self, x):
        """ Returnt the LASSO cost.
        """
        res =  0.5 * np.linalg.norm(self.grad.MX(x) - self.y)**2
        regu = generic_l1_norm(x)
        return res + self.lbda * regu


class DualGap:
    """ Declare the dual-gap cost function.
    """
    def __init__(self, linear_op):
        """ Init.

        Parameters:
        -----------
        linear_op: PISAP wavelet operator,
            the wavelet operator where regularization is done.
        """
        self.linear_op = linear_op

    def __call__(self, x, y):
        """ Return the dual-gap cost.
        """
        return np.linalg.norm(x - self.linear_op.adj_op(y))
