""" Naive dual verification method

Reimplementation of verification from:
https://arxiv.org/abs/1803.06567

Working through this...

Goal is to minimize
      min z_L
s.t.
      z_i = A(x_i)          [for i in 1...L]   [Dual Lambda]
      x_i+1 = ReLU(z_i)     [for i in 1...L-1]

We lagrangify these constraints:

L(x,z, lambda) = z_L + sum_{i=1}^L lambda_2i @ (z_i - Ax_i)
                         + sum_{i=1}^{L-1} Lambda_(2i+1) @ (x_i+1- Relu(Z_i))

And then by weak duality,

Min_xz Max_LambaMu L >= Max (Min L) =: Max (g)

Where g(lambda, mu) can be split by layer and then split by coordinate.
This yields 4 types of terms:

1. -lambda[0] * Linear[0](x_1)
2. (lambda[i-1] - lambda[i]*Linear[i])(x_i)
3. lambda[i]*z_i - lambda[i+1]*relu(z_i)
4. (lambda[L] + 1) * z_L


-------------------------------------
Hence the algorithm is to do the following:
1. Compute pre-activation bounds for the domain
2. Build subroutine to compute g(lambda, mu) in a differentiable fashion
3. Build gradient descent framework (pytorch) to compute dual bounds


"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List

from neural_nets import FFNet, PreactBounds
from abstract_domains import Hyperbox, Zonotope


class NaiveDual:

    def __init__(self, network, input_domain, preact_domain=Hyperbox,
                 prespec_bounds=None, choice='loop'):
        self.network = network
        self.input_domain = input_domain

        # Compute preactivationds (interval analysis)
        self.preact_domain = preact_domain
        if prespec_bounds is not None:
            self.preact_bounds = prespec_bounds
        else:
            self.preact_bounds = PreactBounds(network, input_domain, preact_domain)
            self.preact_bounds.compute()

        self.choice = choice

        # Initialize dual variables
        self.lambda_ = []
        for i, layer in enumerate(self.network):

            self.lambda_.append(torch.zeros_like(self.preact_bounds[i + 1].lbs,
                                                 requires_grad=True))


    def lagrangian(self, x: List[torch.Tensor], verbose=False):
        """ Computes the lagrangian with the current dual variables and
            provided input.
        ARGS:
            x: list of tensors specifying input to each layer (and final layer)
        """
        # Start from the front
        val = 0.0
        for idx, layer in enumerate(self.network):
            layer_val = self.lambda_[idx] @ (x[idx + 1] - layer(x[idx]))
            if verbose:
                print(layer_val)
            val += layer_val
        val += x[-1][0] # Add the output too
        return val

    def lagrange_by_var(self, x: List[torch.Tensor]):
        """
        Prints out the output of each minimization, separated
        by variable. Can help show areas for improvement
        4 types of terms:
        1. -lambda_1 @ A(x1)
        2. sum from i=2 to L of: (-lambda_i @ A + mu_i-1) @ xi
        3. sum from i=1 to L-1 of: lambda_i @ z_i - mu_i @ ReLU(z_i)
        4. (lambda_L + 1) z_L
        """
        total_info = {}
        total_len = len(self.network)
        total_val = 0.0
        for idx, layer in enumerate(self.network):
            pfx = 'x' if (idx % 2 == 0) else 'z'
            varname = '%s:%d' % (pfx, idx // 2)

            val = self.lambda_[idx] @ (-layer(x[idx]))
            if idx > 0:
                val += self.lambda_[idx - 1] @ x[idx]
            total_info[varname] = val
            total_val += val
        output_val = (self.lambda_[-1] + 1) @ x[-1]
        total_info['output'] = output_val
        total_val += output_val
        total_info['total'] = total_val
        return total_info
        # Start from the front


    def argmin(self):
        argmin = []
        for i, bound in enumerate(self.preact_bounds):
            if i in self.network.linear_idxs:
                argmin.append(self.argmin_x_i(i))
            else:

                if self.choice == 'loop':
                    argmin.append(self.argmin_z_i_loop(i))
                else:
                    argmin.append(self.proper_argmin_zi(i))
        return [_.data for _ in argmin]

    def argmin_x_i(self, idx: int):
        """ Gets the argmin for the x_i variable in the lagrangian
            x_i shows up as:

            i == 0:  -Lambda[0] @ A[0](x)
            i > 0 :  Lambda[i+1]@ x -Lambda[i]A[i](x)
        """
        assert idx % 2 == 0 # Generally...
        base_obj = (-self.lambda_[idx].T @ self.network[idx].weight).squeeze()
        if idx > 0:
            base_obj += self.lambda_[idx - 1]
        return self.preact_bounds[idx].solve_lp(base_obj, True)[1]

    def argmin_z_i_loop(self, idx: int):
        """ Gets the argmin for the z_i variable in the lagrangian
            z_i shows up as:
            i < len(network): lambda[idx -1] @ z_i - lambda[idx] @ relu(z_i)
            i = len(network): (1 + lambda[idx]) @ z_i
        """
        assert idx % 2 == 1 # Generally...
        bounds = self.preact_bounds[idx]
        if idx == len(self.network):
            obj = self.lambda_[idx - 1] + 1
            return bounds.solve_lp(obj, True)[1]

        lbs, ubs = bounds.lbs, bounds.ubs

        # Do this in a loop for ease of computation
        argmin = torch.clone(bounds.center)
        rad = bounds.rad
        for coord in range(len(lbs)):
            if lbs[coord] > 0:
                obj = self.lambda_[idx - 1][coord] - self.lambda_[idx][coord]
                argmin[coord] -= torch.sign(obj) * rad[coord]
            elif ubs[coord] < 0:
                obj = self.lambda_[idx - 1][coord]
                argmin[coord] -= torch.sign(obj) * rad[coord]
            else:
                eval_at_l = self.lambda_[idx - 1][coord] * lbs[coord]
                eval_at_u = (self.lambda_[idx - 1][coord] - self.lambda_[idx][coord]) * ubs[coord]

                argmin[coord] = min([(eval_at_l, lbs[coord]),
                                     (eval_at_u, ubs[coord]),
                                     (0.0, 0.0)], key=lambda p: p[0])[1]
        return argmin

    def argmin_z_i(self, idx: int):
        """ Vectorized version of armin_z_i_loop """
        assert idx % 2 == 1 # Generally ...
        bounds = self.preact_bounds[idx]
        if idx == len(self.network):
            obj = self.lambda_[idx - 1] + 1
            return bounds.solve_lp(obj, True)[1]

        lbs, ubs = bounds.lbs, bounds.ubs


        # Create an objective vector for each coordinate that's stable
        stable_obj = torch.zeros_like(lbs)
        on_neurons = (lbs > 0)
        stable_obj[on_neurons] = (self.lambda_[idx - 1][on_neurons] -
                                 self.lambda_[idx][on_neurons])

        off_neurons = (ubs < 0)
        stable_obj[off_neurons] = self.lambda_[idx - 1][off_neurons]

        argmin = bounds.solve_lp(stable_obj, get_argmin=True)[1]

        # And then correct for the unstable ones
        unstable = ~(on_neurons + off_neurons)
        eval_at_l = self.lambda_[idx - 1][unstable] * lbs[unstable]
        eval_at_u = ((self.lambda_[idx - 1][unstable] - self.lambda_[idx][unstable])
                    * ubs[unstable])
        eval_at_0 = torch.zeros_like(eval_at_l)
        min_evals = torch.stack([eval_at_l, eval_at_0, eval_at_u]).min(dim=0)[1]

        argmin[unstable][min_evals == 0] = lbs[unstable][min_evals == 0]
        argmin[unstable][min_evals == 1] = 0.0
        argmin[unstable][min_evals == 2] = ubs[unstable][min_evals == 2]

        return argmin


    def proper_argmin_zi(self, idx: int):
        """ Vectorized version with proper prisming for zonotopes """
        assert idx % 2 == 1
        bounds = self.preact_bounds[idx]
        if idx == len(self.network):
            obj = self.lambda_[idx - 1] + 1
            return bounds.solve_lp(obj, True)[1]
        lbs, ubs = bounds.lbs, bounds.ubs


        # First handle the stable neurons
        stable_obj = torch.zeros_like(lbs)
        on_neurons = (lbs >= 0)
        off_neurons = (ubs < 0)
        stable_obj[on_neurons] = (self.lambda_[idx - 1][on_neurons] -
                                  self.lambda_[idx][on_neurons])
        stable_obj[off_neurons] = self.lambda_[idx - 1][off_neurons]

        argmin = bounds.solve_lp(stable_obj, get_argmin=True)[1]
        #### And then loop to be done

        rad = bounds.rad
        for coord in  (~(on_neurons + off_neurons)).nonzero().squeeze():
            if lbs[coord] > 0:
                obj = self.lambda_[idx - 1][coord] - self.lambda_[idx][coord]
                argmin[coord] -= torch.sign(obj) * rad[coord]
            elif ubs[coord] < 0:
                obj = self.lambda_[idx - 1][coord]
                argmin[coord] -= torch.sign(obj) * rad[coord]
            else:
                eval_at_l = self.lambda_[idx - 1][coord] * lbs[coord]
                eval_at_u = (self.lambda_[idx - 1][coord] - self.lambda_[idx][coord]) * ubs[coord]

                argmin[coord] = min([(eval_at_l, lbs[coord]),
                                     (eval_at_u, ubs[coord]),
                                     (0.0, 0.0)], key=lambda p: p[0])[1]
        return argmin


        ####


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
            optimizer = optim.SGD(self.lambda_, lr=1e-3)
        else:
            optimizer = optimizer(self.lambda_)
        logger = logger or (lambda x: None)

        for step in range(num_steps):
            logger(self)
            optimizer.zero_grad()
            l_val = -self.lagrangian(self.argmin()) # negate to ASCEND
            l_val.backward()
            if (verbose is True or
                (isinstance(verbose, int) and step % verbose == 0)):
                print("Iter %02d | Certificate: %.02f" % (step, -l_val))
            optimizer.step()


        return self.lagrangian(self.argmin())
















