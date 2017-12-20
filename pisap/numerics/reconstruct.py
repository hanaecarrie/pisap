##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
# System import
from __future__ import print_function
import numpy as np
import os
import copy

# Package import
from ..base.image import Image
from ..stats import sigma_mad
from .linear import Identity
from .proximity import SoftThreshold
from .proximity import Positive
from .optimization import CondatVu
from .optimization import FISTA
from .cost import SynthesisCost, AnalysisCost, DualGapCost
from .reweight import cwbReweight
from .reweight import mReweight
from .noise import sigma_mad_sparse


def sparse_rec_condat_vu(
        data, gradient_cls, gradient_kwargs, linear_cls, linear_kwargs,
        std_est=None, std_est_method=None, std_thr=2.,
        mu=1.0e-6, tau=None, sigma=None, relaxation_factor=1.0,
        nb_of_reweights=1, max_nb_of_iter=150, add_positivity=False, atol=1e-4,
        metric_call_period=5, metrics={}, verbose=0):
    """ The Condat-Vu sparse reconstruction with reweightings.

    Parameters
    ----------
    data: ndarray
        the data to reconstruct: observation are expected in Fourier space.
    gradient_cls: class
        a derived 'GradBase' class.
    gradient_kwargs: dict
        the 'gradient_cls' parameters, the first parameter is the data to
        be reconstructed.
    linear_cls: class
        a linear operator class.
    linear_kwargs: dict
        the 'linear_cls' parameters.
    std_est: float (optional, default None)
        the noise std estimate.
        If None use the MAD as a consistent estimator for the std.
    std_est_method: str (optional, default None)
        if the standard deviation is not set, estimate this parameter using
        the mad routine in the image ('image') or in the sparse wavelet
        decomposition ('sparse') domain. The sparse strategy is computed
        at each iteration on the residuals.
    std_thr: float (optional, default 2.)
        use this trehold ewpressed as a number of sigme in  the dual
        proximity operator during the thresholding.
    mu: float (optional, default 1.0e-6)
        regularization hyperparameter
    tau, sigma: float (optional, default None)
        parameters of the Condat-Vu proximal-dual splitting algorithm.
        If None estimates these parameters.
    relaxation_factor: float (optional, default 0.5)
        parameter of the Condat-Vu proximal-dual splitting algorithm.
        If 1, no relaxation.
    nb_of_reweights: int (optional, default 1)
        the number of reweightings.
    max_nb_of_iter: int (optional, default 150)
        the maximum number of iterations in the Condat-Vu proximal-dual
        splitting algorithm.
    add_positivity: bool (optional, default False)
        by setting this option, set the proximity operator to identity or
        positive.
    atol: float (optional, default 1e-4)
        tolerance threshold for convergence.
    metric_call_period: int (default is 5)
        the period on which the metrics are compute.
    metrics: dict, {'metric_name': [metric, if_early_stooping],} (optional)
        the list of desired convergence metrics.
    verbose: int (optional, default 0)
        the verbosity level.
    Returns
    -------
    x_final: Image
        the estimated Condat-Vu primal solution.
    y_final: DictionaryBase
        the estimated Condat-Vu dual solution.
    """
    if verbose > 0:
        print("Starting Condat-Vu primal-dual algorithm.")

    # Check input parameters
    if std_est_method not in (None, "image", "sparse"):
        raise ValueError("Unrecognize std estimation method "
                         "'{0}'.".format(std_est_method))

    # Define the gradient operator
    grad_op = gradient_cls(data, **gradient_kwargs)

    # Define the linear operator
    linear_op = linear_cls(**linear_kwargs)

    img_shape = grad_op.ft_cls.img_shape

    # Define the weights used during the thresholding in the dual domain
    if std_est_method == "image":
        # Define the noise std estimate in the image domain
        if std_est is None:
            std_est = sigma_mad(grad_op.MtX(data))
        weights = linear_op.op(np.zeros(data.shape))
        weights[...] = std_thr * std_est
        reweight_op = cwbReweight(weights, wtype=std_est_method)
        prox_dual_op = SoftThreshold(reweight_op.weights)
        extra_factor_update = sigma_mad_sparse
    elif std_est_method == "sparse":
        # Define the noise std estimate in the image domain
        if std_est is None:
            std_est = 1.0
        weights = linear_op.op(np.zeros(data.shape))
        weights[...] = std_thr * std_est
        reweight_op = mReweight(weights, wtype=std_est_method,
                                thresh_factor=std_thr)
        prox_dual_op = SoftThreshold(reweight_op.weights)
        extra_factor_update = sigma_mad_sparse
    elif std_est_method is None:
        # manual regularization mode
        levels = linear_op.op(np.zeros(img_shape))
        levels[...] = mu
        prox_dual_op = SoftThreshold(levels)
        extra_factor_update = None
        nb_of_reweights = 0

    # Define the Condat Vu optimizer: define the tau and sigma in the
    # Condat-Vu proximal-dual splitting algorithm if not already provided.
    # Check also that the combination of values will lead to convergence.
    norm = linear_op.l2norm(img_shape)
    lipschitz_cst = grad_op.spec_rad
    if sigma is None:
        sigma = 0.5
    if tau is None:
        # to avoid numerics troubles with the convergence bound
        eps = 1.0e-8
        # due to the convergence bound
        tau = 1.0 / (lipschitz_cst/2 + sigma * norm**2 + eps)
        A = 1.0 / tau - sigma * norm ** 2
        B = lipschitz_cst / 2.0
        C = norm**2
        print(A, B, C)

    convergence_test = (
        1.0 / tau - sigma * norm ** 2 >= lipschitz_cst / 2.0)

    if verbose > 0:
        print(" - mu: ", mu)
        print(" - lipschitz_cst: ", lipschitz_cst)
        print(" - tau: ", tau)
        print(" - sigma: ", sigma)
        print(" - rho: ", relaxation_factor)
        print(" - std: ", std_est)
        print(" - 1/tau - sigma||L||^2 >= beta/2: ", convergence_test)
        print("-" * 20)
    if convergence_test == True:
        # Define initial primal and dual solutions
        primal = np.zeros(img_shape, dtype=np.complex)
        dual = linear_op.op(primal)
        dual[...] = 0.0

        # Define the proximity operator
        if add_positivity:
            prox_op = Positive()
        else:
            prox_op = Identity()

        # by default add the lasso cost metric
        lasso = AnalysisCost(data, grad_op, linear_op, mu)
        lasso_cost = {'cost function':{'metric':lasso,
                               'mapping': {'x_new': 'x', 'y_new':None},
                               'cst_kwargs':{},
                               'early_stopping': False,
                               'wind':6,
                               'eps':1.0e-3}}
        metrics.update(lasso_cost)
        # by default add the dual-gap cost metric
        dual_gap = DualGapCost(linear_op)
        dual_gap_cost = {'dual_gap':{'metric':dual_gap,
                                     'mapping': {'x_new': 'x', 'y_new':'y'},
                                     'cst_kwargs':{},
                                     'early_stopping': False,
                                     'wind':6,
                                     'eps':1.0e-3}}
        metrics.update(dual_gap_cost)

        # Define the Condat-Vu optimization method
        opt = CondatVu(x=primal, y=dual, grad=grad_op, prox=prox_op,
                       prox_dual=prox_dual_op, linear=linear_op, sigma=sigma,
                       tau=tau, rho=relaxation_factor, rho_update=None,
                       sigma_update=None, tau_update=None, extra_factor=1.0,
                       extra_factor_update=extra_factor_update,
                       metric_call_period=metric_call_period, metrics=metrics)

        # Perform the first reconstruction
        opt.iterate(max_iter=max_nb_of_iter)

        # Perform reconstruction with reweightings
        # Loop through number of reweightings
        for reweight_index in range(nb_of_reweights):

            # Welcome message
            if verbose > 0:
                print("-" * 10)
                print(" - Reweight: ", reweight_index + 1)
                print("-" * 10)

            # Generate the new weights following reweighting prescription
            if std_est_method == "image":
                reweight_op.reweight(linear_op.op(opt.x_new))
            else:
                std_est = multiscale_sigma_mad(grad_op, linear_op)
                reweight_op.reweight(std_est, linear_op.op(opt.x_new))

            # Update the weights in the dual proximity operator
            prox_dual_op.update_weights(reweight_op.weights)

            # Update the weights in the cost function
            cost_op.update_weights(reweight_op.weights)

            # Perform optimisation with new weights
            opt.iterate(max_iter=max_nb_of_iter)

        linear_op.set_coeff(opt.y_final)

        return Image(data=opt.x_final), linear_op, opt.metrics, opt.is_timeout
        #XXX linear_op.transform -> linear_op for DL
    else:
        message = 'did not pass convergence test'
        return message


def sparse_rec_fista(
        data, gradient_cls, gradient_kwargs, linear_cls, linear_kwargs,
        mu, lambda_init=1.0, max_nb_of_iter=300, atol=1e-4,
        metric_call_period=5, metrics={}, verbose=0):
    """ The Condat-Vu sparse reconstruction with reweightings.

    Parameters
    ----------
    data: ndarray
        the data to reconstruct: observation are expected in Fourier space.
    gradient_cls: class
        a derived 'GradBase' class.
    gradient_kwargs: dict
        the 'gradient_cls' parameters, the first parameter is the data to
        be reconstructed.
    linear_cls: class
        a linear operator class.
    linear_kwargs: dict
        the 'linear_cls' parameters.
    mu: float
       coefficient of regularization.
    lambda_init: float, (default 1.0)
        initial value for the FISTA step.
    max_nb_of_iter: int (optional, default 300)
        the maximum number of iterations in the Condat-Vu proximal-dual
        splitting algorithm.
    atol: float (optional, default 1e-4)
        tolerance threshold for convergence.
    metric_call_period: int (default is 5)
        the period on which the metrics are compute.
    metrics: dict, {'metric_name': [metric, if_early_stooping],} (optional)
        the list of desired convergence metrics.
    verbose: int (optional, default 0)
        the verbosity level.

    Returns
    -------
    x_final: Image,
        the estimated FISTA solution.
    y_final: Dictionary,
        the dictionary transformation estimated FISTA solution
    metrics_list: list of Dict,
        the convergence metrics
    """
    if verbose > 0:
        print("Starting FISTA reconstruction algorithm.")

    # Define the linear operator
    linear_op = linear_cls(**linear_kwargs)

    # Define the gradient operator
    gradient_kwargs["linear_cls"] = linear_op
    grad_op = gradient_cls(data, **gradient_kwargs)
    lipschitz_cst = grad_op.spec_rad

    if verbose > 0:
        print(" - mu: ", mu)
        print(" - lipschitz_cst: ", lipschitz_cst)
        print("-" * 20)

    # Define initial primal and dual solutions
    shape = grad_op.ft_cls.img_shape
    x_init = np.zeros(shape, dtype=np.complex)
    alpha = linear_op.op(x_init)
    alpha[...] = 0.0

    # Define the proximity dual operator
    weights = copy.deepcopy(alpha)
    weights[...] = mu
    prox_op = SoftThreshold(weights)

    # by default add the lasso cost metric
    lasso = SynthesisCost(data, grad_op, mu)
    lasso_cost = {'cost function': {'metric':lasso,
                            'mapping': {'x_new': None, 'y_new':'x'},
                            'cst_kwargs':{},
                            'early_stopping': False,
                            'wind':6,
                            'eps':1.0e-3}}
    metrics.update(lasso_cost)

    opt = FISTA(x=alpha, grad=grad_op, prox=prox_op,
                metric_call_period=metric_call_period, metrics=metrics)

    # Perform the reconstruction
    opt.iterate(max_iter=max_nb_of_iter)

    linear_op.transform.analysis_data = opt.y_final

    return Image(data=opt.x_final), linear_op.transform, opt.metrics
