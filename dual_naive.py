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
import random
import utilities as utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List

from neural_nets import FFNet, PreactBounds
from abstract_domains import Hyperbox, Zonotope
from collections import defaultdict, OrderedDict

class NaiveDual():

    def __init__(self, network, input_domain, preact_domain=Hyperbox,
                 prespec_bounds=None, choice='naive', partition_kwargs=None):
        """
        Main dual object:
        ARGS:
            network: FFNet object, output is 1-dimensional
            input_domain: Abstract Domain object (e.g. Hyperbox object)
            preact_domain: CLASS to be used for preactivation domains
            prespec_bounds: (optional) PreactBounds object, if already precomputed
            choice: string ['naive', 'fw', 'partition'] -- defines how the argmin
                    is computed.
                    'naive' := fastest, the closed form way of doing things
                    'fw' := frank-Wolfe ... isn't really useful lol
                    'partition' := partitions zonotopes by dimension according
                                   to partition_kwargs
            partition_kwargs: (optional) dict with keys like
                partition_style: either 'fixed' or 'random'
                num_partitions: (int) how many partitions to make
        """
        self.network = network
        self.input_domain = input_domain

        # Compute preactivations (interval analysis)
        self.preact_domain = preact_domain
        if prespec_bounds is not None:
            self.preact_bounds = prespec_bounds
        else:
            self.preact_bounds = PreactBounds(network, input_domain, preact_domain)
            self.preact_bounds.compute()


        # Parameters regarding z_i calculation
        self.choice = choice
        self.partition_kwargs = partition_kwargs

        # Initialize dual variables
        self.lambda_ = []
        for i, layer in enumerate(self.network):
            lambda_ = nn.Parameter(torch.zeros_like(self.preact_bounds[i + 1].lbs))
            self.lambda_.append(lambda_)

    def parameters(self):
        return iter(self.lambda_)

    # ======================================================================
    # =           Lagrangian and Argmin methods                            =
    # ======================================================================

    def lagrangian(self, x: List[torch.Tensor], verbose=False,
                   split=False):
        """ Computes the lagrangian with the current dual variables and
            provided input.
        ARGS:
            x: list of tensors specifying input to each layer (and final layer)
        """
        # Start from the front
        val = 0.0
        split_vals = {}
        for idx, layer in enumerate(self.network):
            layer_val = self.lambda_[idx] @ (x[idx + 1] - layer(x[idx]))
            if verbose:
                print(layer_val)

            split_vals[idx] = layer_val
        split_vals[idx + 1] = x[-1][0]
        if split:
            return split_vals
        return sum(split_vals.values())


    def lagrange_by_var(self, x: List[torch.Tensor]):
        """ Same args as lagrangian, but this partitions the output by
            contribution of each (primal) variable
        """

        output = OrderedDict()
        for idx, layer in enumerate(self.network):
            varname = '%s:%d' % ('x' if (idx % 2 == 0) else 'z', idx // 2)
            if idx == 0:
                output[varname] = -self.lambda_[idx] @ layer(x[idx])
            else:
                output[varname] = (self.lambda_[idx - 1] @ x[idx] -
                                   self.lambda_[idx] @ layer(x[idx]))

        output['output'] = (1 + self.lambda_[-1]) * x[-1][0]
        output['total'] = sum(output.values())
        return output



    def lagrange_bounds(self, apx_params=None):
        """ Useful for getting lower/upper bounds for this
            optimization using the MIP.

            apx_params should probably be {'TimeLimit': <int>}

        ARGS:
            apx_params : dict, kwargs to send to zonotope ReLU program.
        RETURNS:
            dict with upper/lower bounds for the output


        """
        total_info = {}
        total_info = {}
        total_len = len(self.network)
        x = self.argmin()
        total_lb = 0.0
        total_ub = 0.0
        for idx, layer in enumerate(self.network):
            pfx = 'x' if (idx % 2 == 0) else 'z'
            varname = '%s:%d' % (pfx, idx // 2)
            if pfx == 'z':
                model = self.apx_mip_zi(idx, apx_params=apx_params)[-1]
                total_info[varname] = model.ObjBound, model.objVal
                total_lb += total_info[varname][0]
                total_ub += total_info[varname][1]
            else:
                total_info[varname] = self.argmin_x_i(idx)[0]
                total_lb += total_info[varname]
                total_ub += total_info[varname]

        output_val = (self.lambda_[-1] + 1) @ x[-1]
        total_info['output'] = output_val
        total_lb += output_val
        total_ub += output_val

        total_info['total_lb'] = total_lb
        total_info['total_ub'] = total_ub
        return total_info


    def argmin(self):
        """ Computes the argmin and returns in a list """
        argmin = []
        for i, bound in enumerate(self.preact_bounds):
            if i in self.network.linear_idxs:
                argmin.append(self.argmin_x_i(i)[1])
            else:
                if self.choice == 'partition':
                    argmin.append(self.partition_zi(i)[1])
                elif self.choice == 'fw':
                    argmin.append(torch.tensor(self.fw_zi(i)[1]))
                elif self.choice == 'simplex':
                    argmin.append(self.simplex_zi(i))
                elif self.choice == 'lbfgsb':
                    argmin.append(self.lbfgsb_zi(i))
                else:
                    #argmin.append(self.naive_argmin_zi(i)[1])
                    argmin.append(self.naive_argmin_zi_noloop(i)[1])
        return [_.data for _ in argmin]



    def dual_ascent(self, num_steps: int, optim_obj=None, verbose=False,
                    logger=None):
        """ Runs the dual ascent process
        ARGS:
            num_steps: number of gradient steps to perform
            optim_obj: Optimizer that modifies the self.lambda_ parameters
            verbose: bool or int -- if int k, prints every k'th iteration
            logger: (optional) function that gets called at every iteration
        RETURNS:
            final dual value
        """

        if optim_obj is None:
            optim_obj = optim.Adam(self.parameters(), lr=1e-3)

        logger = (lambda x: None) if (logger is None) else logger
        for step in range(num_steps):
            logger(self)
            optim_obj.zero_grad()
            l_val = -self.lagrangian(self.argmin()) # negate to ASCEND
            l_val.backward()
            optim_obj.step()

            if verbose and (step % verbose) == 0:
                print("Iter %02d | Certificate: %.02f" % (step, -l_val))


        return self.lagrangian(self.argmin())


    # ============================================================
    # =           Argmin for X_i: this is always exact           =
    # ============================================================

    def argmin_x_i(self, idx: int):
        """ Gets the argmin for the x_i variable in the lagrangian
            x_i shows up as:

            i == 0:  -Lambda[0] @ A[0](x)
            i > 0 :  Lambda[i+1]@ x -Lambda[i]A[i](x)

        RETURNS (min_val, argmin)
        """
        assert idx % 2 == 0 # Generally...
        base_obj = (-self.lambda_[idx].T @ self.network[idx].weight).squeeze()
        if idx > 0:
            base_obj += self.lambda_[idx - 1]
        return self.preact_bounds[idx].solve_lp(base_obj, True)


    # ===========================================================
    # =           Argmin for Z_i: Many choices here             =
    # ===========================================================

    # --------------------------- Naive Z_i ----------------------------
    def naive_argmin_zi(self, idx: int):
        """ Vectorized version with proper prisming for zonotopes
            RETURNS: (min, argmin)
        """
        assert idx % 2 == 1
        bounds = self.preact_bounds[idx]
        if idx == len(self.network):
            obj = self.lambda_[idx - 1] + 1
            return bounds.solve_lp(obj, get_argmin=True)
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

        min_val = self.lambda_[idx -1] @ argmin - self.lambda_[idx] @ torch.relu(argmin)
        return min_val, argmin


    def naive_argmin_zi_noloop(self, idx: int):
        assert idx % 2 == 1
        bounds = self.preact_bounds[idx]
        if idx == len(self.network):
            obj = self.lambda_[idx - 1] + 1
            return bounds.solve_lp(obj, get_argmin=True)

        lin_obj = self.lambda_[idx - 1]
        relu_obj = -self.lambda_[idx]

        # Easy hyperbox case
        if isinstance(bounds, Hyperbox):
            return bounds.solve_relu_program(lin_obj, relu_obj, get_argmin=True)

        # Then partition into stable and unstable cases... and do zonotope
        stable_obj = torch.zeros_like(bounds.lbs)
        on_coords = (bounds.lbs >= 0)
        off_coords = (bounds.ubs < 0)
        ambig_coords = ~(on_coords + off_coords)

        stable_obj[on_coords] = lin_obj[on_coords] + relu_obj[on_coords]
        stable_obj[off_coords] = lin_obj[off_coords]

        argmin = bounds.solve_lp(stable_obj, get_argmin=True)[1]

        # Then do the hyperbox for the rest
        ambig_box = Hyperbox(bounds.lbs[ambig_coords], bounds.ubs[ambig_coords])
        ambig_argmin = ambig_box.solve_relu_program(lin_obj[ambig_coords],
                                                    relu_obj[ambig_coords],
                                                    get_argmin=True)[1]
        argmin[ambig_coords] = ambig_argmin

        opt_val = lin_obj @ argmin + relu_obj @ torch.relu(argmin)

        return opt_val, argmin



    #--------------------------PARTITION STUFF ----------------------------------
    def _default_partition_kwargs(self):
        return {'num_partitions': 10,
                'partition_style': 'random', # vs 'fixed'
                'partitions': None, # subzonos stored here
               }

    def shrink_partitions(self, num_groups):
        if self.partition_kwargs.get('partitions') is None:
            return
        for k in self.partition_kwargs['partitions']:
            new_part = Zonotope.merge_partitions(self.partition_kwargs['partitions'][k], num_groups)
            self.partition_kwargs['partitions'][k] = new_part
        self.partition_kwargs['num_partitions'] = num_groups


    def partition_zi(self, idx: int):
        assert idx % 2 == 1

        # Basic stuff
        bounds = self.preact_bounds[idx]
        if idx == len(self.network): # final one is easy: no relu here
            obj = self.lambda_[idx - 1] + 1
            return bounds.solve_lp(obj, True)
        lbs, ubs = bounds.lbs, bounds.ubs
        dim = lbs.numel()

        # Partition generation
        if self.partition_kwargs is None:
            self.partition_kwargs = self._default_partition_kwargs()
        kwargs = self.partition_kwargs

        # Random partition

        num_parts = kwargs.get('num_partitions', None)
        if num_parts is None:
            part_dim = kwargs.get('partition_dim')
            num_parts = dim // part_dim

        if kwargs['partition_style'] == 'random':
            partitions = bounds.make_random_partitions(num_parts)
        else:
            if kwargs.get('partitions', None) is None:
                kwargs['partitions'] = {}
            if kwargs['partitions'].get(idx, None) is None:
                kwargs['partitions'][idx] = bounds.make_random_partitions(num_parts)
            partitions = kwargs['partitions'][idx]

        # Partition optimization
        c1 = self.lambda_[idx - 1]
        c2 = -self.lambda_[idx]
        if kwargs.get('preproc', None) is None: # preprocess each partition
            preproc = lambda x: x
        outputs = []
        for group, subzono in partitions:
            subzono = preproc(subzono)
            outputs.append(subzono.solve_relu_mip(c1[group], c2[group]))

        # Process output
        argmin = torch.zeros_like(bounds.lbs)
        for (group, subzono), output in zip(partitions, outputs):
            argmin[group] = torch.tensor(output[1], dtype=argmin.dtype)

        min_val = c1 @ argmin + c2 @ torch.relu(argmin)
        return min_val, argmin


    # ------------------------- Approximate MIP ------------------------------
    def apx_mip_zi(self, idx: int, apx_params=None):
        apx_params = apx_params or {'TimeLimit': 2.0}
        assert idx % 2 == 1
        bounds = self.preact_bounds[idx]
        if idx == len(self.network):
            obj = self.lambda_[idx - 1] + 1
            return bounds.solve_lp(obj, True)[0]

        assert isinstance(bounds, Zonotope)
        return bounds.solve_relu_mip(self.lambda_[idx - 1], -self.lambda_[idx],
                                                   apx_params=apx_params)


    # ------------------------------ Frank Wolfe -----------------------------
    def fw_zi(self, idx: int):
        assert idx % 2 == 1
        bounds = self.preact_bounds[idx]
        if idx == len(self.network):
            obj = self.lambda_[idx - 1] + 1
            return bounds.solve_lp(obj, True)

        assert isinstance(bounds, Zonotope)
        zono = bounds
        c1 = self.lambda_[idx - 1]
        c2 = -self.lambda_[idx]

        # Everything is done over y-space!
        def f(y):
            x = zono(y)
            return c1 @ x + c2 @ torch.relu(x)
        def g(y):
            return c1 + torch.relu(zono(y)) * c2
        def s_plus(y):
            # Returns the y that minimizes <g, c + Gy>
            return -torch.sign(g(y) @ zono.generator)

        if True:# rand_init:
            y = torch.rand_like(zono.generator[0]) * 2 - 1.0
        else:
            y = torch.zeros_like(zono.generator[0])

        for step in range(100):# range(num_iter):
            gamma = 2 / float(step + 2.0)
            y = (1 - gamma) * y + gamma * s_plus(y)
        return f(y), zono(y)#, f(y)

    # ------------------------------- SIMPLEX -------------------------------

    def simplex_zi(self, idx: int):
        assert idx % 2 == 1
        bounds = self.preact_bounds[idx]
        if idx == len(self.network):
            obj = self.lambda_[idx - 1] + 1
            return bounds.solve_lp(obj, True)[1]

        assert isinstance(bounds, Zonotope)
        obj, yvals = bounds.solve_relu_simplex(self.lambda_[idx - 1], -self.lambda_[idx])

        # obj2, xvals2 = self.naive_argmin_zi(idx)
        # print(obj2 - obj)

        return bounds(yvals)

    # ------------------------------- L-BFGS-B -------------------------------

    def lbfgsb_zi(self, idx: int):
        assert idx % 2 == 1
        bounds = self.preact_bounds[idx]
        if idx == len(self.network):
            obj = self.lambda_[idx - 1] + 1
            return bounds.solve_lp(obj, True)[1]

        assert isinstance(bounds, Zonotope)
        obj, yvals = bounds.solve_relu_lbfgsb(self.lambda_[idx - 1], -self.lambda_[idx])


        return bounds(yvals)

    # ------------------------- Exact MIP SOLUTION ---------------------------

    def exact_zi(self, idx: int):
        """ Exactly solves the z_i's (uses MIP so very slow)"""
        assert idx % 2 == 1
        bounds = self.preact_bounds[idx]
        if idx == len(self.network):
            obj = self.lambda_[idx - 1] + 1
            return bounds.solve_lp(obj, True)[1]

        if isinstance(bounds, Hyperbox):
            return proper_argmin_zi(idx)
        elif isinstance(bounds, Zonotope):
            obj, xvals, yvals = bounds.solve_relu_mip(self.lambda_[idx - 1], -self.lambda_[idx])
            return xvals













