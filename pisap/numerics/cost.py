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
from sklearn.cluster import k_means
from scipy.ndimage.morphology import binary_closing
from skimage.measure import compare_ssim as _compare_ssim
import matplotlib.pyplot as plt
from pisap.base.utils import min_max_normalize


class AnalysisCost:
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
        regu = np.abs(self.linear_op.op(x)).sum()
        return res + self.lbda * regu


class SynthesisCost:
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
        regu = np.abs(x).sum()
        return res + self.lbda * regu


class DualGapCost:
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


def _preprocess_input(test, ref, mask=None, disp=False):
    """ wrap to the metric

    Parameters:
    -----------
    ref: np.ndarray, the reference image

    test: np.ndarray, the tested image

    mask: np.ndarray, the mask for the ROI

    disp: bool (default False), if True display the mask.

    Notes:
    ------
    Compute the metric only on magnetude.

    Return:
    -------
    ssim: float, the snr
    """
    test = np.abs(test).astype('float64')
    ref = np.abs(ref).astype('float64')
    test = min_max_normalize(test)
    ref = min_max_normalize(ref)
    if (not isinstance(mask, np.ndarray)) and (mask not in ["auto", None]):
        raise ValueError("mask should be None, 'auto' or a np.ndarray,"
                         " got '{0}' instead.".format(mask))
    if mask is None:
        return test, ref, None
    if (not isinstance(mask, np.ndarray)) and (mask == "auto"):
        centroids, mask, _ = k_means(ref.flatten()[:, None], 2)
        if np.argmax(centroids) == 0:
            mask = np.abs(mask-1)
        mask = mask.reshape(*ref.shape)
        mask = binary_closing(mask, np.ones((5, 5)), iterations=4).astype('int')
    if disp:
        plt.matshow(0.5 * (mask + ref), cmap='gray')
        plt.show()
    return test, ref, mask


def ssim(test, ref, mask="auto", disp=False):
    """ Return SSIM

    Parameters:
    -----------
    ref: np.ndarray, the reference image

    test: np.ndarray, the tested image

    mask: np.ndarray, the mask for the ROI

    disp: bool (default False), if True display the mask.

    Notes:
    ------
    Compute the metric only on magnetude.

    Return:
    -------
    ssim: float, the snr
    """
    test, ref, mask = _preprocess_input(test, ref, mask, disp)
    assim, ssim = _compare_ssim(test, ref, full=True)
    if mask is None:
        return assim
    else:
        return (mask * ssim).sum() / mask.sum()


def snr(test, ref, mask=None, disp=False):
    """ Return SNR

    Parameters:
    -----------
    ref: np.ndarray, the reference image

    test: np.ndarray, the tested image

    mask: np.ndarray, the mask for the ROI

    disp: bool (default False), if True display the mask.

    Notes:
    ------
    Compute the metric only on magnetude.

    Return:
    -------
    snr: float, the snr
    """
    test, ref, mask = _preprocess_input(test, ref, mask, disp)
    if mask is not None:
        test = mask * test
        ref = mask * ref
    num = np.mean(np.square(test))
    deno = mse(test, ref)
    return 10.0 * np.log10(num / deno)


def psnr(test, ref, mask=None, disp=False):
    """ Return PSNR

    Parameters:
    -----------
    ref: np.ndarray, the reference image

    test: np.ndarray, the tested image

    mask: np.ndarray, the mask for the ROI

    disp: bool (default False), if True display the mask.

    Notes:
    ------
    Compute the metric only on magnetude.

    Return:
    -------
    psnr: float, the psnr
    """
    test, ref, mask = _preprocess_input(test, ref, mask, disp)
    if mask is not None:
        test = mask * test
        ref = mask * ref
    num = np.max(np.abs(test))
    deno = mse(test, ref)
    return 10.0 * np.log10(num / deno)


def mse(test, ref, mask=None, disp=False):
    """ Return 1/N * |ref - test|_2

    Parameters:
    -----------
    ref: np.ndarray, the reference image

    test: np.ndarray, the tested image

    mask: np.ndarray, the mask for the ROI

    disp: bool (default False), if True display the mask.

    Notes:
    -----
    Compute the metric only on magnetude.

    Return:
    -------
    mse: float, the mse
    """
    test, ref, mask = _preprocess_input(test, ref, mask, disp)
    if mask is not None:
        test = mask * test
        ref = mask * ref
    return np.mean(np.square(test - ref))


def nrmse(test, ref, mask=None, disp=False):
    """ Return NRMSE

    Parameters:
    -----------
    ref: np.ndarray, the reference image

    test: np.ndarray, the tested image

    mask: np.ndarray, the mask for the ROI

    disp: bool (default False), if True display the mask.

    Notes:
    -----
    Compute the metric only on magnetude.

    Return:
    -------
    nrmse: float, the nrmse
    """
    test, ref, mask = _preprocess_input(test, ref, mask, disp)
    if mask is not None:
        test = mask * test
        ref = mask * ref
    num = np.sqrt(mse(test, ref))
    deno = np.sqrt(np.mean((np.square(test))))
    return num / deno
