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
                 prespec_bounds=None, choice='naive', partition_kwargs=None,
                 zero_dual=False):
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
        if not zero_dual:
            self.lambda_ = self.initialize_duals()
        else:
            self.lambda_ = []
            for i, layer in enumerate(self.network):
                if isinstance(layer, nn.ReLU):
                    self.lambda_.append(nn.Parameter(torch.zeros_like(self.preact_bounds[i].lbs)))
                else:
                    self.lambda_.append(None)

    def parameters(self):
        return iter([_ for _ in self.lambda_ if _ is not None])


    def initialize_duals(self):
        """ Dual initialization scheme from KW2017"""
        lambdas = []
        bounds = self.preact_bounds.bounds

        # First compute all the D's and Weights
        diags = []
        weights = []
        for bound, layer in zip(bounds, self.network):

            if isinstance(layer, nn.ReLU):
                lbs, ubs = bound.lbs, bound.ubs
                diag = torch.zeros_like(lbs)
                diag[lbs > 0] = 1
                unc_idxs = (lbs * ubs < 0)
                diag[unc_idxs] = ubs[unc_idxs] / (ubs[unc_idxs] - lbs[unc_idxs])
                weight = None
            else:
                diag = None
                weight = layer.weight
            diags.append(diag)
            weights.append(weight)

        # Then run backwards
        backwards = [weights[-1].squeeze()]
        for i in range(len(diags)-2, -1, -1):
            if i % 2 == 0:
                running = weights[i].T @ backwards[-1]
            else:
                running = diags[i] * backwards[-1]
            backwards.append(running)
        backwards = backwards[::-1]

        # And then attach to lambdas
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.ReLU):
                lambdas.append(nn.Parameter(backwards[i].detach()))
            else:
                lambdas.append(None)

        return lambdas



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


    def lagrange_by_rp(self, x: OrderedDict):
        total = {}
        for i, layer in enumerate(self.network):
            if not isinstance(layer, nn.ReLU):
                continue
            x_b = x[(i, 'B')]
            x_a = x[(i + 2, 'A')]
            if i == 1:
                val = self.lambda_[i + 2] @ self.network[i + 1](F.relu(x_b))
            elif i < len(self.network) - 2:
                val = -self.lambda_[i] @ x_b + self.lambda_[i + 2] @ self.network[i + 1](F.relu(x_b))
            else:
                val = -self.lambda_[i] @ x_b + self.network[i + 1](F.relu(x_b))
            total['P%s' % i] = val
        return total


    def lagrange_bounds(self, apx_params=None):
        total = {}
        for i, layer in enumerate(self.network):
            if not isinstance(layer, nn.ReLU):
                continue
            bounds = self.preact_bounds[i]
            if i == 1:
                c1 = torch.zeros_like(bounds.lbs)
                c2 = self.lambda_[i + 2] @ self.network[i + 1].weight
                bias = self.lambda_[i + 2] @ self.network[i + 1].bias
            elif i < len(self.network) - 2:
                c1 = -self.lambda_[i]
                c2 = self.lambda_[i + 2] @ self.network[i + 1].weight
                bias = self.lambda_[i + 2] @ self.network[i + 1].bias
            else:
                c1 = -self.lambda_[i]
                c2 = self.network[i + 1].weight.squeeze()
                bias = self.network[i + 1].bias.squeeze()
            model = bounds.solve_relu_mip(c1, c2, apx_params=apx_params)[1]
            total['P%s' % i] = (model.ObjBound + bias.item(), model.objVal + bias.item())

        total_lb = sum(_[0] for _ in total.values())
        total_ub = sum(_[1] for _ in total.values())
        total['total_lb'] = total_lb
        total['total_ub'] = total_ub

        return total



    def argmin(self, lambdas=None):
        argmin = OrderedDict()
        for i, layer in enumerate(self.network):
            if isinstance(layer, nn.ReLU):
                i_B, ip1_A = self.argmin_idx(i, lambdas=lambdas)
                argmin[(i, 'B')] = i_B
                argmin[(i+2, 'A')] = ip1_A
        return argmin

    def argmin_idx(self, idx, lambdas=None):
        if self.choice == 'naive':
            i_B, ip1_A = self.argmin_pi_zono(idx, lambdas=lambdas)
        elif self.choice == 'partition':
            i_B, ip1_A = self.argmin_pi_partition(idx, lambdas=lambdas)
        else:
            raise NotImplementedError()
        return i_B, ip1_A




    def _get_coeffs(self, idx: int, lambdas=None):
        """ Gets the coeficients to be used in ReLU programming
        RETURNS: (lin_coeff, relu_coeff)
        lambdas: Non
        """

        lambdas = self.lambda_ if (lambdas is None) else lambdas
        if idx == 1:
            # Special case
            # Min lambda_2 @ A2 (Relu(Z_1A))
            # over the set Z1A in A1(X)
            lin_coeff = torch.zeros_like(self.preact_bounds[idx].lbs)
            relu_coeff = self.network[idx + 1].weight.T @ lambdas[idx + 2]

        elif idx < len(self.network) - 2:
            lin_coeff = -lambdas[idx]
            relu_coeff = self.network[idx + 1].weight.T @ lambdas[idx + 2]
        else:
            lin_coeff = -lambdas[idx]
            relu_coeff = self.network[idx + 1].weight.T
        return lin_coeff, relu_coeff


    def argmin_pi(self, idx: int, lambdas=None):
        """ Gets the argmin for the i^th p_i loop.
            This is indexed where the ReLU's are.
        """
        assert isinstance(self.network[idx], nn.ReLU)
        bounds = self.preact_bounds[idx]
        lbs, ubs = bounds.lbs, bounds.ubs

        # Define the objectives to solve over
        # In general we have
        # Min_Z   c1 @ z + c2 @ relu(z)
        lin_coeff, relu_coeff = self._get_coeffs(idx, lambdas=lambdas)


        argmin = torch.clone(bounds.center)
        rad = bounds.rad
        for coord in range(len(lbs)):
            if lbs[coord] > 0: # relu always on
                obj = lin_coeff[coord] + relu_coeff[coord]
                argmin[coord] -= (torch.sign(obj) * rad[coord]).item()
            elif ubs[coord] < 0:  # relu always off
                obj = lin_coeff[coord]
                argmin[coord] -= (torch.sign(obj) * rad[coord]).item()
            else:
                eval_at_l = lin_coeff[coord] * lbs[coord]
                eval_at_u = (lin_coeff[coord] + relu_coeff[coord]) * ubs[coord]
                argmin[coord] = min([(eval_at_l, lbs[coord]),
                                     (eval_at_u, ubs[coord]),
                                     (0.0, 0.0)], key=lambda p: p[0])[1]

        return (argmin.data, self.network[idx + 1](F.relu(argmin)).data)

    def argmin_pi_zono(self, idx: int, lambdas=None):

        bounds = self.preact_bounds[idx]
        if not isinstance(bounds, Zonotope):
            return self.argmin_pi(idx, lambdas=lambdas)

        assert isinstance(self.network[idx], nn.ReLU)
        lbs, ubs = bounds.lbs, bounds.ubs
        ## LAYERWISE STUFF HERE: Compute the objective vector
        lin_coeff, relu_coeff = self._get_coeffs(idx, lambdas=lambdas)

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


    def _default_partition_kwargs(self):
        return {'num_partitions': 10,
                'partition_style': 'random', # vs 'fixed'
                'partition_rule': 'random',
                'partitions': None, # subzonos stored here
               }

    def shrink_partitions(self, num_groups):
        if self.partition_kwargs.get('partitions') is None:
            return
        for k in self.partition_kwargs['partitions']:
            new_part = Zonotope.merge_partitions(self.partition_kwargs['partitions'][k], num_groups)
            self.partition_kwargs['partitions'][k] = new_part
        self.partition_kwargs['num_partitions'] = num_groups


    def make_partitions(self, idx, lambdas=None):
        kwargs = self.partition_kwargs
        bounds = self.preact_bounds[idx]

        if kwargs['partition_rule'] == 'random':
            return bounds.make_random_partitions(kwargs['num_partitions'])
        elif kwargs['partition_rule'] ==  'min_val':
            c1, c2 = self._get_coeffs(idx, lambdas=lambdas)

            # Score choices to use
            def min_val(zono, i, c1=c1, c2=c2):
                # Takes zonotope and dimension i, outputs float
                half_len = zono.generator[i].abs().sum()
                interval = [zono.center[i] - half_len, zono.center[i] + half_len]
                if interval[0] <= 0 <= interval[1]:
                    interval.append(0.0)
                relu = lambda x: max([x, 0.0])
                return min([c1[i] * el + c2[i] * relu(el) for el in interval])
            return bounds.make_scored_partition(kwargs['num_partitions'], min_val)

        elif kwargs['partition_rule'] == 'minmax_val':
            c1, c2 = self._get_coeffs()
            def minmax_val(zono, i, c1=c1, c2=c2):
                # Takes zonotope and dimension i, outputs float
                half_len = zono.generator[i].abs().sum()
                interval = [zono.center[i] - half_len, zono.center[i] + half_len]
                if interval[0] <= 0 <= interval[1]:
                    interval.append(0.0)
                relu = lambda x: max([x, 0.0])
                return max([c1[i] * el + c2[i] * relu(el) for el in interval]) -\
                       min([c1[i] * el + c2[i] * relu(el) for el in interval])
        else:
            raise NotImplementedError()




    def argmin_pi_partition(self, idx: int, lambdas=None):
        bounds = self.preact_bounds[idx]
        lbs, ubs = bounds.lbs, bounds.ubs
        c1, c2 = self._get_coeffs(idx, lambdas=lambdas)


        # Partition generation
        if self.partition_kwargs is None:
            self.partition_kwargs = self._default_partition_kwargs()
        kwargs = self.partition_kwargs
        if kwargs['partition_style'] == 'random':
            partitions = bounds.make_random_partitions(kwargs['num_partitions'])
        else:
            if kwargs.get('partitions', None) is None:
                kwargs['partitions'] = {}
            if kwargs['partitions'].get(idx, None) is None:
                kwargs

                kwargs['partitions'][idx] = bounds.make_random_partitions(kwargs['num_partitions'])
            partitions = kwargs['partitions'][idx]

        # Partition optimization
        if kwargs.get('preproc', None) is None: # preprocess each partition
            preproc = lambda x: x
        outputs = []
        for group, subzono in partitions:
            subzono = preproc(subzono)
            outputs.append(subzono.solve_relu_mip(c1[group], c2[group]))

        argmin = torch.zeros_like(bounds.lbs)
        for (group, subzono), output in zip(partitions, outputs):
            argmin[group] = torch.tensor(output[1], dtype=argmin.dtype)

        min_val = c1.squeeze() @ argmin.squeeze() + c2.squeeze() @ torch.relu(argmin).squeeze()
        return (argmin.data , self.network[idx + 1](F.relu(argmin)).data)




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
            optim_obj = optim.Adam(self.parameters(),
                                  lr=1e-3)
        logger = logger or (lambda x: None)
        for step in range(num_steps):
            logger(self)
            optim_obj.zero_grad()
            l_val = -self.lagrangian(self.argmin()) # negate to ASCEND
            l_val.backward()
            if verbose and (step % verbose) == 0:
                print("Iter %02d | Certificate: %.02f" % (step, -l_val))
            optim_obj.step()


        return self.lagrangian(self.argmin())



    def proximal_method(self, num_steps, num_inner_steps, eta_k=None, optim_obj=None, verbose=False,
                        logger=None):
        """ Runs the proximal method for dual ascent updates -- maybe this is better?
        ARGS:
            num_steps: int - number of outer steps to perform
            num_inner_steps: int - number of inner steps for each outer step
            eta_k : (int -> float) -  function that maps step k to eta value
            optim_obj: ???
            verbose : bool - prints stuff out if true
            logger: (optional) function that gets called at every iteration, takes self as arg
        """

        # Initialize things:
        # Assume dual variables already initialized
        primal = self.argmin()

        convex_combo = lambda old, new, gamma: gamma * new + (1 - gamma) * old

        if isinstance(eta_k, (float, int)): #number becomes constant fxn
            eta_k = lambda k: eta_k


        lambdas = [_ for _ in self.lambda_] # shallow copy here
        for step in range(num_steps):
            inner_lambdas = [_ for _ in lambdas]
            for inner_step in range(num_inner_steps):
                # Do block coordinate updates
                for idx, layer in enumerate(self.network):
                    if not isinstance(layer, nn.ReLU):
                        continue
                    if i > 1: # zonotopes don't need i=1 case
                        inner_lambdas[idx] = lambdas[idx] + (primal[(idx, 'A')] - primal[(idx, 'B')]) / eta(idx)
                    i_B, ip1_A = self.argmin_idx(idx, lambdas=inner_lambdas)
                    step_size = self._get_opt_fw_step(primal, i_B, ip1_A, eta_k, idx)

                    primal[(idx, 'B')] = convex_combo(primal[idx], i_B, step_size)
                    primal[(idx + 2, 'A')] = convex_combo(primal[idx], ip1_A, step_size)

            lambdas = self._update_fw_lambdas(primals, lambdas, eta_k)


    def _get_opt_fw_step(self, primal, eta_k, idx):
        # Gets gamma in [0,1] for val
        return 0.5
        pass

    def _update_fw_lambdas(self, primals, lambdas, eta_k):

        pass



