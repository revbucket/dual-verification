""" File that has neural nets with easy support for abstract interpretations
    and dual verification techniques
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import utilities as utils
from lp_relaxation import LPRelax

from abstract_domains import Hyperbox, Zonotope, Polytope
from decomposition_plnn_bounds.plnn_bounds.proxlp_solver.solver import SaddleLP



class FFNet(nn.Module):
    """ Wrapper for FeedForward Neural Networks.
        Handles networks with architectures composed of sequential
        operations of:
        - Linear operators : FullyConnected, Convolutional
        - Nonlinear layers : ReLU
    """

    SUPPORTED_LINS = [nn.Linear, nn.Conv2d, nn.AvgPool2d]
    SUPPORTED_NONLINS = [nn.ReLU]

    # ======================================================================
    # =           Constructor methods and helpers                          =
    # ======================================================================


    def __init__(self, net, dtype=torch.float):
        super(FFNet, self).__init__()
        self.net = net
        self.dtype = dtype
        self.shapes = None
        self._support_check()


        # Auxiliary attributes
        self.linear_idxs = set([i for i, layer in enumerate(self.net)
                                if type(layer) in self.SUPPORTED_LINS])
        self.nonlin_idxs = set([i for i, layer in enumerate(self.net)
                                if type(layer) in self.SUPPORTED_NONLINS])


    def __getitem__(self, idx):
        return self.net[idx]

    def __len__(self):
        return len(self.net)

    def _support_check(self):
        for layer in self.net:
            assert any([isinstance(layer, tuple(self.SUPPORTED_LINS)),
                        isinstance(layer, tuple(self.SUPPORTED_NONLINS))])



    @classmethod
    def relu_net(cls, sizes, bias=False, dtype=torch.float):
        # Creates a network with ReLU nonlinearities
        seq_list = []
        for idx in range(len(sizes) - 1):
            shape_pair = (sizes[idx], sizes[idx + 1])
            seq_list.append(nn.Linear(*shape_pair, bias=bias, dtype=dtype))
            if idx < len(sizes) -1:
                seq_list.append(nn.ReLU())
        return cls(nn.Sequential(*seq_list), dtype=dtype)


    def binarize(self, i:int, j:int):
        """ Turns this net into a binary classifier by constructing a new
            final linear layer, subtracting class i from j, and making a
            new instance that now has _scalar_ outputs
        ARGS:
            i: positive class
            j: negative class
            e.g. f(x)_i - f(x)_j
        """
        final_linear = self.net[-1]
        new_linear_layer = nn.Linear(in_features=final_linear.in_features,
                                     out_features=1,
                                     device=final_linear.weight.device)
        weight_shape = new_linear_layer.weight.shape
        new_linear_layer.weight.data = (final_linear.weight[i] -
                                        final_linear.weight[j]).data.view(weight_shape)
        new_linear_layer.bias.data = (final_linear.bias[i] -
                                      final_linear.bias[j]).data.view(1)

        new_net = nn.Sequential(*list(self.net[:-1]) + [new_linear_layer])
        return FFNet(new_net, self.dtype)


    # ====================================================================
    # =           Evaluation/Forward pass stuff                          =
    # ====================================================================

    def forward(self, x, pfx=None):
        shapes = []
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                x = torch.flatten(x, 1)
            if self.shapes is None:
                shapes.append(x.shape[1:])
                layer.input_shape = shapes[-1]
            x = layer(x)

        if len(shapes) > 0:
            self.shapes = shapes
        return x

    def _reshape(self, x):
        return x.view(-1, *self.net[0].input_shape)


    def pfx_forward(self, x, k):
        # Evaluates only the FIRST k layers (ReLU's included) of the net
        x = self._reshape(x)
        return self.net[:k](x)

    def sfx_forward(self, x):
        # Evaluates only the LAST k layers (ReLU's included) of the net
        # (assumes input is shaped correctly here)
        return self.net[-k:](x)



# ===========================================================================
# =                             Preactivation Bounds                        =
# ===========================================================================


class PreactBounds:
    def __init__(self, network: FFNet, input_range: Hyperbox, abstract_domain):
        """ Object which holds and computes preactivation bounds"""
        self.network = network
        self.input_range = input_range
        self.abstract_domain = abstract_domain
        self.bounds = []
        self.computed = False

    def __getitem__(self, idx):
        return self.bounds[idx]

    def __len__(self):
        return len(self.bounds)

    def compute_polytope(self):
        """ Special logic for polytopes... """
        poly = Polytope(self.input_range, lp_everything=True)
        for layer in self.network:
            poly.map_layer(layer)
        self.bounds = [v for k,v in poly.bounds.items()]
        self.polytope = poly
        self.computed = True

    def compute_lpr(self):
        lpr = LPRelax(self.network, self.input_range)
        lpr.setup(lp_everything=True)
        lpr.compute()
        self.lpr = lpr
        for k in sorted(lpr.input_bounds.keys()):
            self.bounds.append(Hyperbox(*lpr.input_bounds[k].T))
        self.computed = True

    def compute(self):
        if self.abstract_domain == Polytope:
            return self.compute_lpr()
            #return self.compute_polytope()


        with torch.no_grad():
            if self.abstract_domain == Zonotope:
                self.input_range = self.input_range.as_zonotope()

            self.bounds.append(self.input_range)
            for layer in self.network:
                self.bounds.append(self.bounds[-1].map_layer(layer))
        self.computed = True


class KWBounds(PreactBounds):
    def __init__(self, network: FFNet, input_range: Hyperbox):
        super(KWBounds, self).__init__(network, input_range, Hyperbox)

    def compute(self):
        # test_input is a hyperbox

        input_shape = self.network[0].input_shape

        domain = torch.stack([self.input_range.lbs.view(input_shape), self.input_range.ubs.view(input_shape)], dim=-1)
        kw_net = SaddleLP([lay for lay in self.network])

        kw_net.define_linear_approximation(domain, force_optim=False, no_conv=False)
        intermediate_lbs = kw_net.lower_bounds
        intermediate_ubs = kw_net.upper_bounds


        # And then need to fix to be in my setup:
        bounds = [self.input_range]
        running_idx = 1
        for layer in self.network:
            prev_layer = bounds[-1]
            if isinstance(layer, nn.ReLU):
                bounds.append(Hyperbox(torch.relu(prev_layer.lbs), torch.relu(prev_layer.ubs)))
                continue
            elif isinstance(layer, nn.Linear):
                bounds.append(Hyperbox(intermediate_lbs[running_idx],
                                       intermediate_ubs[running_idx]))
                running_idx += 1

        bounds.append(Hyperbox(intermediate_lbs[-1], intermediate_ubs[-1]))


        self.bounds = bounds
        self.computed = True
        return self


class CROWN(PreactBounds):
    def __init__(self, network: FFNet, input_range: Hyperbox):
        raise NotImplementedError("This is broken I think")
        super(CROWN, self).__init__(network, input_range, Hyperbox)

    def compute(self):
        self.bounds.append(self.input_range)
        for i in range(len(self.network)):
            self.bounds.append(self._compute_ith_layer(i))

    @utils.no_grad
    def _compute_ith_layer(self, i):
        if i == 0 or isinstance(self.network[i], nn.ReLU):
            return self.bounds[i].map_layer(self.network[i])

        # Set up lambdas/mus
        lambdas, mus = [], []
        for j in range(i):
            if isinstance(self.network[j], nn.ReLU):
                lb, ub = self.bounds[i].lbs, self.bounds[i].ubs
                lambda_ = torch.zeros_like(lb)
                mu = torch.zeros_like(lb)
                lambda_[lb >= 0] = 1.0
                unc = (lb * ub) < 0
                lambda_[unc] = ub[unc] / (ub[unc] - lb[unc])
                mu[unc] = -lambda_[unc] * lb[unc]

                lambdas.append(lambda_)
                mus.append(mu)
            else:
                lambdas.append(None)
                mus.append(None)

        # Do the backprop
        w = self.network[i].weight
        lower_bias = self.network[i].bias
        upper_bias = self.network[i].bias
        for j in range(i -1, -1, -1):
            layer = self.network[j]
            if isinstance(layer, nn.Linear):
                upper_bias += w @ layer.bias
                lower_bias += w @ layer.bias
                w = w @ layer.weight

            else:
                upper_bias += torch.relu(w.data) @ mus[j]
                lower_bias += -torch.relu(-w.data) @ mus[j]
                w = w * lambdas[j].view(1, -1)

        # Compute the final outputs
        final_center = w @ self.input_range.center
        final_rad = w.abs() @ self.input_range.rad

        final_lbs = final_center - final_rad + lower_bias
        final_ubs = final_center + final_rad + upper_bias
        return Hyperbox(final_lbs, final_ubs)
