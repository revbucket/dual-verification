""" Dual via lagrangian decomposition

Reimplementation of verification from:

Goal is to minimize
        min z_L^A

s.t.
        z_1^B = A(X)   #input constraints
        \\ x_i+1 = relu(z_i^B)    <we call this P_i(z_i^B, z_i+1^A)
        \\ z_i+1^A = A(x_i+1)     For i in [1, ... L-1]
        z_i^A = z_i^B             For i in [2, ... L-1]
Where we only lagrangify the final constraint

This yields lagrangian
L(x, zA, zB, lambda ) = zA_L + sum_{i=2}^{L-1} lambda_i @ (zA_i - zB_i)

And then weak duality holds so
min_xz max_lambda L >= max_lambda min_xz L

And then we can split by layer so... we have three types of constraints
1. lambda_1 @ (zA_1)   (over zA_1 in A(X))
2. -lambda_i @ (zB_i) + lambda_{i+1} @ (zA_i+1) (over P_i) [i ... L-2]
3. (1 + lambda) zA_L

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List
from collections import OrderedDict
from neural_nets import FFNet, PreactBounds
from abstract_domains import Hyperbox, Zonotope


class DecompDual:
    def __init__(self, network, input_domain, preact_domain=Hyperbox,
                 prespec_bounds=None, clever_zono=False):
        self.network = network
        self.input_domain = input_domain

        # Compute preactivations (interval analysis)
        self.preact_domain = preact_domain
        if prespec_bounds is not None:
            self.preact_bounds = prespec_bounds
        else:
            self.preact_bounds = PreactBounds(network, input_domain, preact_domain)
            self.preact_bounds.compute()

        self.clever_zono = clever_zono
        # Initialize dual variables
        self.lambda_ = []
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.ReLU):
                self.lambda_.append(torch.zeros_like(self.preact_bounds[i].lbs,
                                                     requires_grad=True))
            else:
                self.lambda_.append(None)


    def lagrangian(self, x: OrderedDict, by_var=False):
        """ Computes the lagrangian with the current dual variables and
            provided input.
        ARGS:
            x: OrderedDict of primal variables
        """
        # Start from the front:

        total = {}
        for i, layer in enumerate(self.network):
            if not isinstance(layer, nn.ReLU) or i == 1:
                continue
            total[i] = self.lambda_[i] @ (x[(i, 'A')] - x[(i, 'B')])
        total['output'] = x[(len(self.network), 'A')].item()
        total['total'] = sum(total.values())
        if by_var:

            return total

        return sum(total.values()) - total['total']


    def lagrange_by_var(self, x: OrderedDict):
        return self.lagrangian(x, by_var=True)


    def argmin(self):
        argmin = OrderedDict()
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.ReLU):
                if self.clever_zono:
                    i_B, ip1_A = self.argmin_pi_zono2(i)
                else:
                    i_B, ip1_A = self.argmin_pi(i)
                argmin[(i, 'B')] = i_B
                argmin[(i+2, 'A')] = ip1_A
        return argmin


    def argmin_pi(self, idx: int):
        """ Gets the argmin for the i^th p_i loop.
            This is indexed where the ReLU's are.
        """
        assert isinstance(self.network[idx], nn.ReLU)
        bounds = self.preact_bounds[idx]
        lbs, ubs = bounds.lbs, bounds.ubs

        # Define the objectives to solve over
        # In general we have
        # Min_Z   c1 @ z + c2 @ relu(z)
        if idx == 1:
            # Special case
            # Min lambda_2 @ A2 (Relu(Z_1A))
            # over the set Z1A in A1(X)
            lin_coeff = torch.zeros_like(lbs)
            relu_coeff = self.network[idx + 1].weight.T @ self.lambda_[idx + 2]

        elif idx < len(self.network) - 2:
            lin_coeff = -self.lambda_[idx]
            relu_coeff = self.network[idx + 1].weight.T @ self.lambda_[idx + 2]
        else:
            lin_coeff = -self.lambda_[idx]
            relu_coeff = self.network[idx + 1].weight.T


        argmin = torch.clone(bounds.center)
        rad = bounds.rad
        for coord in range(len(lbs)):
            if lbs[coord] > 0: # relu always on
                obj = lin_coeff[coord] + relu_coeff[coord]
                argmin[coord] -= torch.sign(obj) * rad[coord]
            elif ubs[coord] < 0:  # relu always off
                obj = lin_coeff[coord]
                argmin[coord] -= torch.sign(obj) * rad[coord]
            else:
                eval_at_l = lin_coeff[coord] * lbs[coord]
                eval_at_u = (lin_coeff[coord] + relu_coeff[coord]) * ubs[coord]
                argmin[coord] = min([(eval_at_l, lbs[coord]),
                                     (eval_at_u, ubs[coord]),
                                     (0.0, 0.0)], key=lambda p: p[0])[1]

        return (argmin.data, self.network[idx + 1](F.relu(argmin)).data)

    def argmin_pi_zono2(self, idx: int):

        bounds = self.preact_bounds[idx]
        if not isinstance(bounds, Zonotope):
            return self.argmin_pi(idx)

        assert isinstance(self.network[idx], nn.ReLU)
        lbs, ubs = bounds.lbs, bounds.ubs
        ## LAYERWISE STUFF HERE: Compute the objective vector
        if idx == 1:
            # Special case
            # Min lambda_2 @ A2 (Relu(Z_1A))
            # over the set Z1A in A1(X)
            lin_coeff = torch.zeros_like(lbs)
            relu_coeff = self.network[idx + 1].weight.T @ self.lambda_[idx + 2]

        elif idx < len(self.network) - 2:
            lin_coeff = -self.lambda_[idx]
            relu_coeff = self.network[idx + 1].weight.T @ self.lambda_[idx + 2]
        else:
            lin_coeff = -self.lambda_[idx]
            relu_coeff = self.network[idx + 1].weight.T


        ## Now do the more clever zonotope thing
        #  We separate into fixed-relu and non-fixed settings
        #  In the fixed case, we create a projected zonotope and
        #  solve the optimization for those coordinates

        fixed_idxs = ((lbs * ubs) >= 0).nonzero().squeeze()
        off_subidxs = (ubs <= 0)[fixed_idxs].nonzero().squeeze()
        fixed_lin = lin_coeff[fixed_idxs]
        fixed_relu = torch.clone(relu_coeff[fixed_idxs])
        fixed_relu[off_subidxs] = 0.
        fixed_obj = (fixed_lin + fixed_relu).view(-1)
        expanded_fixed_obj = torch.zeros_like(lbs)
        expanded_fixed_obj[fixed_idxs] = fixed_obj

        _, fixed_argmin = bounds.solve_lp(expanded_fixed_obj, get_argmin=True)

        ## Now do the coordinate-wise thing for non-fixed neurons
        non_fixed_idxs = ((lbs * ubs) < 0).nonzero().squeeze()
        argmin = torch.clone(bounds.center)
        rad = bounds.rad
        for coord in non_fixed_idxs:
            eval_at_l = lin_coeff[coord] * lbs[coord]
            eval_at_u = (lin_coeff[coord] + relu_coeff[coord]) * ubs[coord]
            argmin[coord] = min([(eval_at_l, lbs[coord]),
                                 (eval_at_u, ubs[coord]),
                                 (0.0, 0.0)], key=lambda p: p[0])[1]

        argmin[fixed_idxs] = fixed_argmin[fixed_idxs]

        return (argmin.data, self.network[idx + 1](F.relu(argmin)).data)







    def dual_ascent(self, num_steps: int, optimizer=None, verbose=False,
                    logger=None):
        """ Runs the dual ascent process
        ARGS:
            num_steps: number of gradient steps to perform
            optimizer: if optimizer is not None, is a partially applied
                       function that just takes the tensors to optimize as
                       an arg
        RETURNS:
            final dual value
        """
        if optimizer is None:
            optimizer = optim.SGD([_ for _ in self.lambda_ if _ is not None],
                                  lr=1e-3)
        else:
            optimizer = optimizer([_ for _ in self.lambda_ if _ is not None])
        logger = logger or (lambda x: None)
        for step in range(num_steps):
            logger(self)
            optimizer.zero_grad()
            l_val = -self.lagrangian(self.argmin()) # negate to ASCEND
            l_val.backward()
            if verbose:
                print("Iter %02d | Certificate: %.02f" % (step, -l_val))
            optimizer.step()


        return self.lagrangian(self.argmin())

