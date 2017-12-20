##########################################################################
# XXX - Copyright (C) XXX, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
import copy
import progressbar
import numpy as np
import time
import signal
from ..base.observable import Observable, MetricObserver
from datetime import timedelta
from datetime import datetime

class FISTA(Observable):
    """ Fast Iterative Shrinkage-Thresholding Algorithm.
    """
    def __init__(self, x, grad, prox, lbda_init=1.0, metric_call_period=5,
                 metrics={}):
        """ Init.

        Parameters
        ----------
        x: np.ndarray
            Initial guess for the primal variable
        grad: class
            Gradient operator class
        prox: class
            Proximity operator class
        lbda_init: float
            Initial value of the relaxation parameter
        metric_call_period: int, (default is 5)
            the period on which the metrics are computed.
        metrics: dict, {'metric_name': [metric, if_early_stooping],} (optional)
            the list of desired convergence metrics.
        ref: np.ndarray (default None)
            image reference if given.
        """
        self.x_old = x
        self.z_old = copy.deepcopy(self.x_old)
        self.z_new = copy.deepcopy(self.z_old)
        self.grad = grad
        self.prox = prox
        self.lbda = self.t_old = lbda_init
        self.metric_call_period = metric_call_period
        Observable.__init__(self, ["cv_metrics"])
        for name, dic in metrics.iteritems():
            observer = MetricObserver(name, dic['metric'],
                                      dic['mapping'],
                                      dic['cst_kwargs'],
                                      dic['early_stopping'],
                                      dic['wind'],
                                      dic['eps'])
            self.add_observer("cv_metrics", observer)

    def update(self):
        """ Compute one iteration of the FISTA algorithm.
        """
        self.grad.get_grad(self.z_old)
        y_old = self.z_old - self.grad.inv_spec_rad * self.grad.grad
        self.x_new = self.prox.op(y_old,
                                  extra_factor=self.grad.inv_spec_rad)
        self.z_new = self.x_old + self.lbda * (self.x_new - self.x_old)
        self.params_update()
        self.x_old = copy.deepcopy(self.x_new)
        self.z_old = copy.deepcopy(self.z_new)

    def params_update(self):
        """ Update the parameters of convergence.
        """
        self.t = 0.5 * (1 + np.sqrt(1 + 4 * self.t_old ** 2))
        self.lbda = 1 + (self.t_old - 1) / self.t
        self.t_old = self.t

    def iterate(self, max_iter=150):
        """ Compute max_iter iterations of the FISTA algorithm.

        Parameters
        ----------
        max_iter: int, (default 150)
           maximum number of iterations.
        """
        bar = progressbar.ProgressBar()
        for i in bar(range(max_iter)):
            self.update()
            if i % self.metric_call_period == 0:
                kwargs = {'x_new': self.grad.linear_cls.adj_op(self.z_new),
                          'y_new': self.z_new, 'idx':i}
                self.notify_observers('cv_metrics', **kwargs)
                if self.any_convergence_flag():
                    print("-------> early-stopping done")
                    break
        self.retrieve_outputs()

    def any_convergence_flag(self):
        """ Return if any matrices values matched the convergence criteria.
        """
        return any([obs.converge_flag for obs in self._observers['cv_metrics']])

    def retrieve_outputs(self):
        """ Declare the outputs of the algorithms as attributes: x_final,
        metrics.
        """
        metrics = {}
        for obs in self._observers['cv_metrics']:
            metrics[obs.name] = obs.retrieve_metrics()
        self.x_final = self.grad.linear_cls.adj_op(self.z_new)
        self.y_final = self.z_new
        self.metrics = metrics

class CondatVu(Observable):
    """ Condat-Vu primal dual optimisation algorithm.
    """
    def __init__(self, x, y, grad, prox, prox_dual, linear, sigma, tau,
                 rho=1.0, rho_update=None, sigma_update=None, tau_update=None,
                 extra_factor=1.0, extra_factor_update=None,
                 metric_call_period=5, metrics={}):
        """ Init.

        Parameters
        ----------
        x: np.ndarray
            Initial guess for the primal variable
        y: np.ndarray
            Initial guess for the dual variable
        grad: class
            Gradient operator class
        prox: class
            Proximity primal operator class
        prox_dual: class
            Proximity dual operator class
        linear: class
            Linear operator class
        rho: float
            Relaxation parameter
        sigma: float
            Proximal dual parameter
        tau: float
            Proximal primal paramater
        rho_update:
            Relaxation parameter update method
        sigma_update:
            Proximal dual parameter update method
        tau_update:
            Proximal primal parameter update method
        extra_factor_update:
            Extra factor passed to the dual proximity operator update
        metric_call_period: int, (default is 5)
            the period on which the metrics are computed.
        metrics: dict, {'metric_name': [metric, if_early_stooping],} (optional)
            the list of desired convergence metrics.
        """
        self.x_old = x
        self.x_new = np.copy(self.x_old)
        self.y_old = y
        self.grad = grad
        self.prox = prox
        self.prox_dual = prox_dual
        self.linear = linear
        self.rho = rho
        self.sigma = sigma
        self.tau = tau
        self.rho_update = rho_update
        self.sigma_update = sigma_update
        self.tau_update = tau_update
        self.extra_factor = 1.0
        self.extra_factor_update = extra_factor_update
        self.metric_call_period = metric_call_period
        self.is_timeout = False
        Observable.__init__(self, ["cv_metrics"])
        for name, dic in metrics.iteritems():
            observer = MetricObserver(name, dic['metric'],
                                      dic['mapping'],
                                      dic['cst_kwargs'],
                                      dic['early_stopping'],
                                      dic['wind'],
                                      dic['eps'])
            self.add_observer("cv_metrics", observer)

    def update(self):
        """ Compute one iteration of the Condat-Vu algorithm.
        """
        self.grad.get_grad(self.x_old)
        self.params_update()
        x_tmp = (self.x_old - self.tau * self.grad.grad - self.tau *
                  self.linear.adj_op(self.y_old))
        x_prox = self.prox.op(x_tmp)
        y_tmp = (self.y_old + self.sigma *
                self.linear.op(2 * x_prox - self.x_old))
        y_prox = (y_tmp - self.sigma *
                  self.prox_dual.op(y_tmp / self.sigma,
                                    extra_factor=(1/self.sigma)))
        self.x_new = self.rho * x_prox + (1 - self.rho) * self.x_old
        self.y_new = self.rho * y_prox + (1 - self.rho) * self.y_old
        np.copyto(self.x_old, self.x_new)
        self.y_old = copy.deepcopy(self.y_new)


    def params_update(self):
        """ Update the parameters of convergence.
        """
        if self.rho_update is not None:
            self.rho = self.rho_update(self.rho)
        if self.sigma_update is not None:
            self.sigma = self.sigma_update(self.sigma)
        if self.tau_update is not None:
            self.tau = self.tau_update(self.tau)
        if self.extra_factor_update is not None:
            self.extra_factor = self.extra_factor_update(self.extra_factor)

    def iterate(self, max_iter=150):
        """ Compute max_iter iterations of the Condat-Vu algorithm.

        Parameters
        ----------
        max_iter: int, (default 150)
           maximum number of iterations.
        """
        bar = progressbar.ProgressBar()
        wait_until = datetime.now() + timedelta(minutes=10)
        for i in bar(range(max_iter)):
            if wait_until < datetime.now():
                self.is_timeout = True
                #print("-------> timeout")
                #break
            self.update()
            if wait_until < datetime.now():
                self.is_timeout = True
                #print("-------> timeout")
                #break
            if i % self.metric_call_period == 0:
                kwargs = {'x_new': self.x_new, 'y_new':self.y_new, 'idx':i}
                self.notify_observers('cv_metrics', **kwargs)
                if self.any_convergence_flag():
                    #print("-------> early-stopping done")
                    #break
            if wait_until < datetime.now():
                self.is_timeout = True
                #print("-------> timeout")
                #break
        self.retrieve_outputs()

    def any_convergence_flag(self):
        """ Return if any matrices values matched the convergence criteria.
        """
        return any([obs.converge_flag for obs in self._observers['cv_metrics']])

    def retrieve_outputs(self):
        """ Declare the outputs of the algorithms as attributes: x_final,
        y_final, metrics.
        """
        metrics = {}
        for obs in self._observers['cv_metrics']:
            metrics[obs.name] = obs.retrieve_metrics()
        self.x_final = self.x_new
        self.y_final = self.y_new
        self.metrics = metrics
