""" Dual via operator splitting (DeepSplit)

The relaxed version of this problem becomes...

min (c @ x_L)
s.t.
    y_k = x_k       k=0,...L-1
    (yk,zk) in S_k  k=0,...L-1
    x_k+1 = z_k     k=0,...L-1
    x0 in domain

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List
from collections import OrderedDict
from neural_nets import FFNet, PreactBounds
from abstract_domains import Hyperbox, Zonotope


class DeepSplit:
    def __init__(self, network, input_domain, cvec, preact_domain=Hyperbox,
                 hyperparams=None):

        self.network = network
        self.input_domain = input_domain
        self.cvec # c-vector that we dot with x_L to form objective

        # Compute preactivations
        self.preact_domain = preact_domain
        self.preact_bounds = PreactBounds(network, input_domain, preact_domain)
        self.preact_bounds.compute()

        # Initialize dual variable
        # Need to maintain an x,y,z
        # as well as dual variables lambda, mu
        self.state = self.init_state()
        self.affine_cache = self.cache_affines()

        # Hyperparameters for optimization
        if hyperparams is None:
            hyperparams = {'rho': 10.0,
                           'mu': 2.0,
                           'tau': 2.0}
        self.hyperparams = hyperparams


    def optimize(self, num_iter, residual_balance=True):
        pass



    def init_state(self):
        """ Initialize the dual and intermed variables and put them in a dict"""

        center = self.preact_bounds[0].center()

        xs, ys, zs = [center], [], [] # primal vars
        lambdas, mus = [], [] # dual vars
        for layer in self.network:
            # Primals...
            ys.append(xs[-1].clone())
            zs.append(layer(ys[-1]).clone())
            xs.append(zs[-1].clone())

            # Duals...
            lambdas.append(torch.zeros_like(ys[-1], requires_grad=True))
            mus.append(torch.zeros_like(xs[-1], requires_grad=True))

        return {'xs': xs,
                'ys': ys,
                'zs': zs,
                'lambdas': lambdas,
                'mus': mus}

    def cache_affines(self):
        """ Computes and stores the inverse like (I+W^TW)^-1 for every weight matrix"""

        self.affines = []
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear):
                eye = torch.eye(layer.in_features, dtype=layer.weight.dtype)
                inv = torch.linalg.inv(eye + layer.weight.T @ layer.weight)
                self.affines.append(inv)
            else:
                self.affines.append(None)


    def update_x(self):
        """ Modifies the state dict to update x
        x' = argmin_x L
           = {
              proj_X(y_0-lambda_0)                        if k = 0
              argmin xl + rho/2 * ||xl - zl-1 + ul-1||^2  if k = l
              1/2(y_k - lambda_k + z_k-1 - u_k-1)         otherwise
           }

        """
        xs, ys, zs = self.state['xs'], self.state['ys'], self.state['zs']
        lambdas, mus = self.state['lambdas'], self.state['mus']
        for k, el in enumerate(xs):
            if k == 0:
                el.data = self.preact_bounds[0].project(ys[0] - lambdas[0])
            elif k == len(xs) - 1:
                el.data = zs[-1] - mus[-1] - self.cvec / self.rho
            else:
                el.data = 0.5 * (y[k] - lambdas[k] + zs[k-1] - mus[k-1])


    def update_yz(self):
        """ Modifies the state dict to update y,z
        In general, this is Proj_Sk( x_k + lambda k, x_k+1 + mu_k)

        """
        xs, ys, zs = self.state['xs'], self.state['ys'], self.state['zs']
        lambdas, mus = self.state['lambdas'], self.state['mus']

        for k, layer in enumerate(self.network):
            if isinstance(layer, nn.Linear):
                inner = (self.xs[k] + self.lambdas[k] +
                         layer.weight.T @ (self.xs[k+1] + self.mus[k] - layer.bias))
                ys[k].data = self.affines[k] @ inner
                zs[k].data = layer(ys[k])

            elif isinstance(layer, nn.ReLU):
                self.project_relu(k)
            else:
                raise NotImplementedError


    def project_relu(self, k):
        y, z = self.state['ys'][k], self.state['zs'][k]
        bounds = self.preact_bounds[k]
        lbs, ubs = bounds.lbs, bounds.ubs

        on_idxs = (lbs >= 0)
        off_idxs = (ubs < 0)

        # Always-on coordinates
        y[on_idxs].data = z[on_idxs].data = torch.clamp((y + z)[on_idxs]/2.0,
                                                        lbs[on_idxs], ubs[on_idxs])

        # Always-off coordinates
        y.data[off_idxs] = torch.clamp(y[off_idxs], lbs[off_idxs], lbs[on_idxs])
        z.data[off_idxs] = 0.0

        # Projection for uncertains...
        unc_idxs = (~(on_idxs + off_idxs)).nonzero()
        for idx in unc_idxs:
            y1 = z1 = torch.clamp((y[idx] + z[idx]) /2.0, 0, ubs[idx])


            s = ubs[idx]  / (ubs[idx] - lbs[idx])
            y2 = torch.clamp((y[idx] + s * (z[idx] + s * lbs[idx])) / (s ** 2 + 1),
                             lbs[idx], ubs[idx])
            z2 = (s * (y[idx] - lbs[idx]) + s ** 2 * (z[idx])) / (s ** 2 + 1)

            y3 = torch.clamp(y[idx], lbs[idx], 0.0)
            z3 = 0.0

            y[idx], z[idx] = min([(y1, z1), (y2, z2), (y3, z3)],
                                  key=lambda y_,z_: (y[idx] - y_) **2 + (z[idx] - z) ** 2)

        return



    def update_duals(self):
        """ Modifies the state dict to update lambda, mu """
        xs, ys, zs = self.state['xs'], self.state['ys'], self.state['zs']
        lambdas, mus = self.state['lambdas'], self.state['mus']

        for k in range(len(lambdas)):
            lambdas[k].data += (xs[k] - ys[k])
            mus[k].data += (xs[k] - zs[k])


    def residuals(self):
        pass


