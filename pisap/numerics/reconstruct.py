##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
#
#:Author: Samuel Farrens <samuel.farrens@gmail.com>
##########################################################################

# System import
from __future__ import print_function
import numpy as np
import time
import copy
import pickle
from scipy.linalg import norm

# Package import
import pisap
from pisap.stats import sigma_mad
from pisap.base.dictionary import Identity
from .proximity import SoftThreshold
from .proximity import Positive
from .optimization import ForwardBackward
from .optimization import Condat
from .optimization import GenForwardBackward
from .reweight import cwbReweight
from .reweight import mReweight
from .cost import costFunction
from .noise import sigma_mad_sparse


def sparse_rec_condat_vu(
        data, gradient_cls, gradient_kwargs, linear_cls, linear_kwargs,
        std_est=None, std_est_method=None, std_thr=2., mu=1.0e-6, tau=None,
        sigma=None, relaxation_factor=1.0, nb_of_reweights=1, max_nb_of_iter=150,
        add_positivity=False, atol=1e-4, verbose=0, report=True):
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
    verbose: int (optional, default 0)
        the verbosity level.
    report: bool (optional, default True)
        if true generate a pickle report file
    Returns
    -------
    x_final: Image
        the estimated Condat-Vu primal solution.
    y_final: DictionaryBase
        the estimated Condat-Vu dual solution.
    """
    # Welcome message
    start = time.clock()
    if verbose > 0:
        print("-" * 20)
        print("Starting Condat-Vu proximal-dual splitting reconstruction "
              "algorithm.")
        linear_op = linear_cls(**linear_kwargs)
        print("The linear op used:\n{0}".format(linear_op.op(np.zeros(data.shape))))

    # Check input parameters
    if std_est_method not in (None, "image", "sparse"):
        raise ValueError("Unrecognize std estimation method "
                         "'{0}'.".format(std_est_method))

    # Define the gradient operator
    grad_op = gradient_cls(data, **gradient_kwargs)

    # Define the linear operator
    linear_op = linear_cls(**linear_kwargs)

    # Define the weights used during the thresholding in the dual domain
    if std_est_method == "image":
        # Define the noise std estimate in the image domain
        if std_est is None:
            std_est = sigma_mad(grad_op.MtX(data))
        weights = linear_op.op(np.zeros(data.shape))
        weights.set_constant_values(values=(std_thr * std_est))
        reweight_op = cwbReweight(weights, wtype=std_est_method)
        prox_dual_op = SoftThreshold(reweight_op.weights)
        extra_factor_update = sigma_mad_sparse
    elif std_est_method == "sparse":
        # Define the noise std estimate in the image domain
        if std_est is None:
            std_est = 1.0
        weights = linear_op.op(np.zeros(data.shape))
        weights.set_constant_values(values=(std_thr * std_est))
        reweight_op = mReweight(weights, wtype=std_est_method,
                                thresh_factor=std_thr)
        prox_dual_op = SoftThreshold(reweight_op.weights)
        extra_factor_update = sigma_mad_sparse
    elif std_est_method is None:
        # manual regularization mode
        levels = linear_op.op(np.zeros(data.shape))
        levels.set_constant_values(values=mu)
        prox_dual_op = SoftThreshold(levels)
        extra_factor_update = None
        nb_of_reweights = 0

    # Define the Condat Vu optimizer: define the tau and sigma in the
    # Condat-Vu proximal-dual splitting algorithm if not already provided.
    # Check also that the combination of values will lead to convergence.
    norm = linear_op.l2norm(data.shape)
    lipschitz_cst = grad_op.spec_rad
    if sigma is None:
        sigma = 0.5
    if tau is None:
        # to avoid numerics troubles with the convergence bound
        eps = 1.0e-8
        # due to the convergence bound
        tau = 1 / (lipschitz_cst/2 + sigma * norm + eps)

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

    # Define initial primal and dual solutions
    primal = np.zeros(data.shape, dtype=np.complex)
    dual = linear_op.op(primal)
    dual.set_constant_values(values=0.0)

    # Define the proximity operator
    if add_positivity:
        prox_op = Positive()
    else:
        prox_op = Identity()

    # Define the cost operator
    cost_op = costFunction(
        y=data,
        grad=grad_op,
        wavelet=linear_op,
        weights=levels,
        lambda_reg=mu,
        mode="lasso",
        window=2,
        print_cost=verbose > 0,
        tolerance=atol,
        output="plot_condat.jpg",
        positivity=False)

    # Define the Condat-Vu optimization method
    opt = Condat(
        x=primal,
        y=dual,
        grad=grad_op,
        prox=prox_op,
        prox_dual=prox_dual_op,
        linear=linear_op,
        cost=cost_op,
        rho=relaxation_factor,
        sigma=sigma,
        tau=tau,
        extra_factor_update=extra_factor_update,
        auto_iterate=False)

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
            reweight_op.reweight(opt.extra_factor, linear_op.op(opt.x_new))

        # Update the weights in the dual proximity operator
        prox_dual_op.update_weights(reweight_op.weights)

        # Update the weights in the cost function
        cost_op.update_weights(reweight_op.weights)

        # Perform optimisation with new weights
        opt.iterate(max_iter=max_nb_of_iter)

    # Finish message
    end = time.clock()

    if report:
        filename = "condat_report_" + time.strftime("%m_%d__%H_%M_%S") + ".pkl"
        to_dump = {'nb_iter': max_nb_of_iter,
                   'mu': mu,
                   'cost_list': np.array(cost_op.cost_list),
                   'regu_list': np.array(cost_op.regu_list),
                   'res_list': np.array(cost_op.res_list),
                   'x':opt.x_final,
                   'y':opt.y_final,
                   'x_':linear_op.adj_op(opt.y_final),
                   'y_':linear_op.op(opt.x_final),
                   'data': data,
                   }
        with open(filename, "wb") as pfile:
            pickle.dump(to_dump, pfile)

    if verbose > 0:
        #cost_op.plot_cost()
        print("-" * 20)
        print(" - Final iteration number: ", cost_op.iteration)
        print(" - Final log10 cost value: ", np.log10(cost_op.cost))
        print(" - Converged: ", opt.converge)
        print(" - Execution time: ", end - start, " seconds")

    return pisap.Image(data=opt.x_final), opt.y_final


def sparse_rec_fista(
        data, gradient_cls, gradient_kwargs, linear_cls, linear_kwargs,
        mu, lambda_init=1.0, max_nb_of_iter=300, atol=1e-4, verbose=0,
        report=True):

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
    verbose: int (optional, default 0)
        the verbosity level.
    report: bool (optional, default True)
        if true generate a pickle report file

    Returns
    -------
    x_final: Image
        the estimated FISTA solution.
    """
   # Welcome message
    start = time.clock()
    if verbose > 0:
        print("-" * 20)
        print("Starting FISTA reconstruction algorithm.")
        print("argmin_alpha |Ft*L*alpha - y|_2^2 + mu * |alpha|_1")

    # Define the linear operator
    linear_op = linear_cls(**linear_kwargs)


    # Define the gradient operator
    gradient_kwargs['linear_cls'] = linear_op
    grad_op = gradient_cls(data, **gradient_kwargs)

    lipschitz_cst = grad_op.spec_rad

    if verbose > 0:
        print(" - mu: ", mu)
        print(" - lipschitz_cst: ", lipschitz_cst)
        print("-" * 20)

    # Define initial primal and dual solutions
    shape = (grad_op.ft_cls.img_size, grad_op.ft_cls.img_size)
    x_init = np.zeros(shape, dtype=np.complex) # grad_op.MtX(data)
    alpha = linear_op.op(x_init)
    alpha.set_constant_values(values=0.)

    # Define the proximity dual operator
    weights = copy.deepcopy(alpha)
    weights.set_constant_values(values=mu) # re-double check
    prox_op = SoftThreshold(weights)

    # Define the cost operator
    cost_op = costFunction(
        y=data,
        grad=grad_op,
        wavelet=Identity(),
        weights=weights,
        lambda_reg=mu,
        mode="lasso",
        window=2,
        output="cost_fista.jpg",
        print_cost=verbose > 0,
        tolerance=atol,
        positivity=False)

    # Define the FISTA optimization method
    opt = ForwardBackward(
        x=alpha,
        grad=grad_op,
        prox=prox_op,
        cost=cost_op,
        lambda_init = lambda_init,
        lambda_update=None,
        use_fista=True,
        auto_iterate=False)

    # Perform the reconstruction
    opt.iterate(max_iter=max_nb_of_iter)

    # Finish message
    end = time.clock()

    if report:
        filename = "fista_report_" + time.strftime("%m_%d__%H_%M_%S") + ".pkl"
        to_dump = {'delta'
                   'nb_iter': max_nb_of_iter,
                   'mu': mu,
                   'cost_list': np.array(cost_op.cost_list),
                   'regu_list': np.array(cost_op.regu_list),
                   'res_list': np.array(cost_op.res_list),
                   'x':linear_op.adj_op(opt.x_final),
                   'data': data,
                   }
        with open(filename, "wb") as pfile:
            pickle.dump(to_dump, pfile)

    if verbose > 0:
        #cost_op.plot_cost()
        print("-" * 20)
        print(" - Final iteration number: ", cost_op.iteration)
        print(" - Final log10 cost value: ", np.log10(cost_op.cost))
        print(" - Converged: ", opt.converge)
        print(" - Execution time: ", end - start, " seconds")

    return pisap.Image(data=linear_op.adj_op(opt.x_final))
