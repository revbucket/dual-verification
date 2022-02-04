""" Dual decompose solved the way Alessandro et al solve it
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from typing import List
from collections import OrderedDict
from neural_nets import FFNet, PreactBounds, KWBounds, BoxInformedZonos
from abstract_domains import Hyperbox, Zonotope
import utilities as utils
from partitions import PartitionGroup
from decomposition_plnn_bounds.plnn_bounds.proxlp_solver.solver import SaddleLP


class DecompDual:
    def __init__(self, network, input_domain, preact_domain=Hyperbox,
                 choice='naive', partition=None, preact_bounds=None,
                 zero_dual=True, primal_mip_kwargs=None, mip_start=None,
                 num_ub_iters=1): # TODO: implement KW initial dual bounds

        self.network = network
        self.input_domain = input_domain

        self.preact_bounds = (preact_bounds if (preact_bounds is not None) else
                              PreactBounds(network, input_domain, preact_domain).compute())

        self.choice = choice
        self.partition = partition
        self.primal_mip_kwargs = primal_mip_kwargs

        self.mip_start = mip_start
        self.num_ub_iters = num_ub_iters
        # Initialize duals

        self.rhos = self.init_duals(zero_dual)


    # ==============================================================================
    # =           Helper methods                                                   =
    # ==============================================================================
    def init_duals(self, zero_dual):
        if zero_dual:
            return self._zero_duals()
        else:
            return self._kw_init_duals()

    def _zero_duals(self):


        rhos = OrderedDict()
        for idx, layer in enumerate(self.network):
            if isinstance(layer, nn.ReLU):

                rhos[idx] = nn.Parameter(torch.zeros_like(self.preact_bounds[idx].lbs))
        return rhos

    def _kw_init_duals(self):
        # Stealing this code because I'm lazy
        rhos = OrderedDict()
        domain = torch.stack([self.input_domain.lbs.view(self.network[0].input_shape),
                              self.input_domain.ubs.view(self.network[0].input_shape)], dim=-1)
        coeffs_idx = len([_ for _ in self.network if isinstance(_, nn.ReLU)]) + 1
        coeffs = {coeffs_idx: torch.Tensor([[-1.0]]).to(self.network[0].weight.device)}
        intermed_net = SaddleLP([lay for lay in utils.add_flattens(self.network)])
        intermed_net.set_solution_optimizer('init', None)
        intermed_net.define_linear_approximation(domain, force_optim=True, no_conv=False,
                                                         override_numerical_errors=True)

        intermed_net.set_initialisation('KW')
        kw_rhos = intermed_net.decomposition.initial_dual_solution(intermed_net.weights, coeffs,
                                            intermed_net.lower_bounds, intermed_net.upper_bounds).rhos


        running_idx = 0
        for idx, layer in enumerate(self.network):
            if isinstance(layer, nn.ReLU):
                rhos[idx] = nn.Parameter(kw_rhos[running_idx].flatten().data)
                running_idx += 1
        return rhos


    def parameters(self):
        return iter(self.rhos.values())


    def _get_coeffs(self, idx: int, rhos=None):
        rhos = self.rhos if (rhos is None) else rhos

        # subproblem indices are {0, 1, 3, 5,...,}
        # Now with added bias term...

        if idx == 0:
            # Special case
            # Min -rho[1] * Aff(x_input)
            lin_coeff = torch.zeros_like(self.preact_bounds[0].lbs)
            if isinstance(self.network[0], nn.Linear):
                out_coeff = (rhos[1] @ self.network[0].weight).squeeze()
            elif isinstance(self.network[0], nn.Conv2d):
                conv = self.network[0]
                output_shape = utils.conv_output_shape(conv)
                # Need to compute output padding here
                transpose_shape = utils.conv_transpose_shape(conv)
                output_padding = (conv.input_shape[1] - transpose_shape[1],
                                  conv.input_shape[2] - transpose_shape[2])
                out_coeff = F.conv_transpose2d(rhos[1].view((1,) + output_shape),
                                               conv.weight, None, conv.stride, conv.padding,
                                               output_padding, conv.groups, conv.dilation).flatten()
            else:
                raise NotImplementedError()

        elif idx < len(self.network) - 2:
            lin_coeff = -rhos[idx]
            if isinstance(self.network[idx + 1], nn.Linear):
                out_coeff = (rhos[idx + 2] @ self.network[idx + 1].weight).squeeze()
            elif isinstance(self.network[idx + 1], nn.Conv2d):
                conv = self.network[idx + 1]
                output_shape = utils.conv_output_shape(conv)
                transpose_shape = utils.conv_transpose_shape(conv)
                output_padding = (conv.input_shape[1] - transpose_shape[1],
                                  conv.input_shape[2] - transpose_shape[2])
                out_coeff = F.conv_transpose2d(rhos[idx + 2].view((1,) + output_shape),
                                               conv.weight, None, conv.stride, conv.padding,
                                               output_padding, conv.groups, conv.dilation).flatten()

        else:
            lin_coeff = -rhos[idx]
            out_coeff = self.network[idx + 1].weight.squeeze()
        return lin_coeff, out_coeff


    # ===========================================================================
    # =           Lagrangian compute methods                                    =
    # ===========================================================================

    def lagrangian(self, primals=None, rhos=None):
        # Compute lagrangian with argmins here
        total = {}
        rhos = self.rhos if (rhos is None) else rhos
        primals = primals if (primals is not None) else self.get_primals(rhos=rhos)[1]

        for idx, layer in enumerate(self.network):
            if not isinstance(layer, nn.ReLU):
                continue
            x_a = primals[(idx, 'A')]
            x_b = primals[(idx, 'B')]
            total[idx] = rhos[idx] @ (x_a - x_b)
        # And final value is just the last variable
        total[len(self.network)] = primals[(len(self.network), 'A')].item()

        return sum(total.values())




    def lagrangian_mip_bounds(self, rhos=None, time_limit=None):
        """ Returns bounds on the dual given the dual vars (rhos)
        In a dict like:
            {subproblem_idx: (subproblem_lb, subproblem_Ub, subproblem_time)}
        """
        rhos = rhos if (rhos is not None) else self.rhos
        apx_params = {'MIPFocus': 3}
        if time_limit is not None:
            apx_params['TimeLimit'] = time_limit

        total = {}
        for idx, layer in enumerate(self.network):
            start_time = time.time()
            clock = lambda: time.time() - start_time
            if idx == 0:
                val = self.get_0th_primal(rhos=rhos)[0]
                total[idx] = (val, val, clock())
            elif isinstance(layer, nn.ReLU):
                model = self.get_ith_primal_mip(idx, rhos, apx_params=apx_params, return_model=True)
                total[idx] = (model.ObjBound, model.objVal, clock())

        total_lbs = sum(_[0] for _ in total.values()).data.item()
        total_ubs = sum(_[1] for _ in total.values()).data.item()
        total_time = sum(_[2] for _ in total.values())
        total['total'] = (total_lbs, total_ubs, total_time)
        return total


    def get_primals(self, rhos=None):
        # Returns like (total_val_dict, argmin_dict)
        argmin = OrderedDict()
        total_vals = {}
        for idx, layer in enumerate(self.network):
            if idx == 0:
                # Solve linear case here
                opt_val, primal_in, primal_out = self.get_0th_primal(rhos=rhos)
                argmin[(1,'A')] = primal_out

            elif isinstance(layer, nn.ReLU):
                opt_val, x_b, x_a = self.get_ith_primal(idx, rhos=rhos)
                argmin[(idx, 'B')] = x_b
                argmin[(idx + 2, 'A')] = x_a
            else:
                continue
            total_vals[idx] = opt_val

        return total_vals, argmin


    def get_ith_primal(self, idx, rhos=None):
        # Outputs here are (opt_val, argmin, next_layer_argmin)
        rhos = rhos if (rhos is not None) else self.rhos

        method_dict = {'naive': self.get_ith_primal_naive,
                       'partition': self.get_ith_primal_partition,
                       'simplex': self.get_ith_primal_simplex,
                       'simplex_partition': self.get_ith_primal_simplex_partition,
                       'lbfgsb': self.get_ith_primal_lbfgsb,
                       'lbfgsb_partition': self.get_ith_primal_lbfgsb_partition,
                       'exact': self.get_ith_primal_mip
                      }

        ub_methods = {'simplex', 'simplex_partition', 'lbfgsb', 'lbfgsb_partition'}


        if isinstance(self.preact_bounds[idx], Hyperbox): # Hyperbox means do naive case
            return method_dict['naive'](idx, rhos)

        if self.choice in ub_methods:
            min_val, min_argmin_0, min_argmin_1 = float('inf'), None, None
            for i in range(self.num_ub_iters):
                val, argmin_0, argmin_1 = method_dict[self.choice](idx, rhos)
                if val < min_val:
                    min_val = val
                    min_argmin_0 = argmin_0
                    min_argmin_1 = argmin_1
            return min_val, min_argmin_0, min_argmin_1
        else:
            return method_dict[self.choice](idx, rhos)


    def dual_ascent(self, num_steps, optim_obj=None, verbose=True, logger=None, iter_start=0):
        """ Runs dual ascent for num_steps, using the optim_obj specified
            logger is a fxn that takes (self, step#) as args
        """
        optim_obj = optim_obj if (optim_obj is not None) else optim.Adam(self.parameters(), lr=1e-3)
        logger = (lambda self, iter_num: None) if (logger is None) else logger
        last_time = time.time()
        for step in range(iter_start, iter_start + num_steps):
            optim_obj.zero_grad()
            loss_val = -1 * self.lagrangian()
            loss_val.backward()
            if verbose and (step % verbose) == 0:
                print("Iter %02d | Certificate: %.02f  | Time: %.02f" % (step, -loss_val, time.time() - last_time))
                last_time = time.time()
            logger(self, step)
            optim_obj.step()

        return self.lagrangian()

    @utils.no_grad
    def manual_dual_ascent(self, num_steps, verbose=True, iter_start=0,
                           optim_params=None):
        """ Runs dual_ascent for num_steps, but manually does ADAM """
        if optim_params is None:
            optim_params = {'initial_eta': 1e-3,
                            'final_eta': 1e-6,
                            'betas': (0.9, 0.999)}


        def get_subg():
            # Get gradient as dual residuals
            primals = self.get_primals(rhos=self.rhos)[1]
            dual_subg = OrderedDict()
            for k in self.rhos:
                dual_subg[k] = primals[(k, 'A')] - primals[(k, 'B')]
            return dual_subg

        def get_stepsize(t):
            return optim_params['initial_eta'] + t / float(num_steps) * (optim_params['final_eta'] - optim_params['initial_eta'])

        # Initialize adam parameters
        m_i = {k: torch.zeros_like(p) for k, p in self.rhos.items()}
        v_i = {k: torch.zeros_like(p) for k, p in self.rhos.items()}


        # Update dual params using ADAM rule
        # Follow pseudocode directly
        beta1, beta2 = optim_params['betas']
        eps = 1e-6
        last_time = time.time()
        for step in range(1, num_steps + 1):
            step = float(step)
            stepsize = get_stepsize(step)
            subg = get_subg()

            m_i = {k: beta1 * v + (1 - beta1) * subg[k] for k, v in m_i.items()}
            v_i = {k: beta2 * v + (1 - beta2) * torch.pow(subg[k], 2) for k, v in v_i.items()}
            mhat = {k: v / (1 - beta1 ** step) for k, v in m_i.items()}
            vhat = {k: v / (1 - beta2 ** step) for k, v in v_i.items()}
            for k, v in self.rhos.items():
                v.data += stepsize * mhat[k] / (torch.sqrt(vhat[k])+ eps) # 'PLUS' here for dual ascent!

            if verbose and (step % verbose) == 0:
                loss_val = self.lagrangian()
                print("Iter %02d | Certificate: %.02f  | Time: %.02f" % (step, loss_val, time.time() - last_time))
                last_time = time.time()





    # ================================================================================
    # =                            PRIMAL SOLVERS                                    =
    # ================================================================================
    # ALL PRIMAL SOLVERS OUTPUT (value, argmin, armin_next_layer)
    #------------------------------- 0th layer is always an LP -----------------------------

    def _next_layer_relu_out(self, idx, x):
        layer = self.network[idx + 1]
        return layer(x.relu().view(layer.input_shape).unsqueeze(0)).flatten()

    def get_0th_primal(self, rhos):
        """ Solves min_x -rho[1] @ layers[0](x) over x in input
                    (actually) min_x -rho[1] @ (Wx + b)
                               so need to subtract -rho[1] @b
            RETURNS: (x, layers[0](x))
        """
        _, out_coeff = self._get_coeffs(0, rhos=rhos)
        bound = self.preact_bounds[0]
        opt_val, x = bound.solve_lp(out_coeff, get_argmin=True)
        return opt_val, x, self.network[0](x.view(self.network[0].input_shape).unsqueeze(0)).flatten()


    def get_ith_primal_naive(self, idx, rhos):
        """ Solves min_z rho[idx]@z - rho[idx+2] @ Aff(relu(z))
            for z in bounds[z]
            RETURNS (z, layers[idx + 2](relu(z)))
        """
        lin_coeff, relu_coeff = self._get_coeffs(idx, rhos=rhos)
        bound = self.preact_bounds[idx]
        if isinstance(bound, Hyperbox):
            opt_val, x = bound.solve_relu_program(lin_coeff, relu_coeff, get_argmin=True)
        else:
            # Then partition into stable and unstable cases... and do zonotope
            stable_obj = torch.zeros_like(bound.lbs)
            on_coords = (bound.lbs >= 0)
            off_coords = (bound.ubs < 0)
            ambig_coords = ~(on_coords + off_coords)

            stable_obj[on_coords] = lin_coeff[on_coords] + relu_coeff[on_coords]
            stable_obj[off_coords] = lin_coeff[off_coords]

            opt_val_0, argmin = bound.solve_lp(stable_obj, get_argmin=True)

            # Then do the hyperbox for the rest
            ambig_box = Hyperbox(bound.lbs[ambig_coords], bound.ubs[ambig_coords])
            opt_val_1, ambig_argmin = ambig_box.solve_relu_program(lin_coeff[ambig_coords],
                                                                   relu_coeff[ambig_coords],
                                                                   get_argmin=True)
            argmin[ambig_coords] = ambig_argmin
            x = argmin.data
            opt_val = opt_val_0 + opt_val_1

        return opt_val, x, self._next_layer_relu_out(idx, x)


    #---------------------------------- PARTITION STUFF ----------------------------

    def gather_partitions(self):
        """ Use this method to access partition object.
            If partition is None, makes a basic partition (2d)
            If partition doesn't have zonos attached, attaches them
            otherwise returns PartitionGroup
        """
        # Use this method to access partition object.
        base_zonotopes = {i: self.preact_bounds[i] for i, bound in enumerate(self.preact_bounds)
                          if i not in self.network.linear_idxs}

        input_shapes = [l.input_shape for l in self.network.net]

        if self.partition is None:
            box_info = None
            if isinstance(self.preact_bounds, BoxInformedZonos):
                box_info = {(2 *i - 1): box for i, box in enumerate(self.preact_bounds.box_range[1:], start=1)}

            partition = PartitionGroup(base_zonotopes, style='fixed_dim', partition_rule='random',
                                       save_partitions=True, save_models=False, partition_dim=2,
                                       use_crossings=True, box_info=box_info, input_shapes=input_shapes)
            self.partition = partition

        if self.partition.base_zonotopes is None:
            self.partition.attach_zonotopes(base_zonotopes)
        return self.partition


    def merge_partitions(self, partition_dim=None, num_partitions=None, copy_obj=True):

        self.partition.save_models = True # Generally want to do this to save time later
        self.partition = self.gather_partitions().merge_partitions(partition_dim=partition_dim,
                                                                   num_partitions=num_partitions,
                                                                   copy_obj=copy_obj)



    def get_ith_primal_partition(self, idx: int, rhos):
        # Basic stuff:
        bounds = self.preact_bounds[idx]
        assert isinstance(bounds, Zonotope)
        c1, c2 = self._get_coeffs(idx, rhos=rhos)


        opt_val, x = self.gather_partitions().relu_program(idx, c1, c2,
                                                           gurobi_params=self.primal_mip_kwargs,
                                                           start=self.mip_start)
        return opt_val, x.data, self._next_layer_relu_out(idx, x.data)


    # -------------------------- SIMPLEX -------------------------------------------
    def get_ith_primal_simplex(self, idx: int, rhos):
        bounds = self.preact_bounds[idx]
        assert isinstance(bounds, Zonotope)

        c1, c2 = self._get_coeffs(idx, rhos=rhos)
        opt_val, yvals = bounds.solve_relu_simplex(c1, c2)
        x = bounds(yvals)
        return opt_val, x.data, self._next_layer_relu_out(idx, x.data)

    def get_ith_primal_simplex_partition(self, idx: int, rhos):
        bounds = self.preact_bounds[idx]
        assert isinstance(bounds, Zonotope)
        c1, c2 = self._get_coeffs(idx, rhos=rhos)
        # HACK: write the right answer for 2d partitions
        partition = self.gather_partitions()
        if partition.partition_dim == 2 or (isinstance(partition.partition_dim, dict) and partition.partition_dim[idx] == 2):
            opt_val, x = partition.relu_program(idx, c1, c2, gurobi_params=self.primal_mip_kwargs,
                                          start=self.mip_start)
        else:
            opt_val, x = partition.relu_program_simplex(idx, c1, c2)
        return opt_val, x.data, self._next_layer_relu_out(idx, x.data)


    # ------------------------ L-BFGS-B --------------------------------------

    def get_ith_primal_lbfgsb(self, idx: int, rhos):
        bounds = self.preact_bounds[idx]
        c1, c2 = self._get_coeffs(idx, rhos=rhos)
        opt_val, yvals = bounds.solve_relu_lbfgsb(c1, c1)
        x = bounds(yvals)
        return opt_val, x.data, self._next_layer_relu_out(idx, x.data)


    def get_ith_primal_lbfgsb_partition(self, idx: int, rhos):
        bounds = self.preact_bounds[idx]
        assert isinstance(bounds, Zonotope)

        c1, c2 = self._get_coeffs(idx, rhos=rhos)
        opt_val, x = self.gather_partitions().relu_program_lbfgsb(idx, c1, c2)
        return opt_val, x.data, self._next_layer_relu_out(idx, x.data)

    # ---------------------- Frank Wolfe ----------------------------------------

    def get_ith_primal_fw(self, idx: int, rhos):
        bounds = self.preact_bounds[idx]
        c1, c2 = self._get_coeffs



    # ---------------------- Other MIP-y methods -------------------------------------
    def get_ith_primal_mip(self, idx: int, rhos, apx_params=None, return_model=False):
        bounds = self.preact_bounds[idx]
        assert isinstance(bounds, Zonotope)
        apx_params = apx_params if (apx_params is not None) else self.primal_mip_kwargs
        c1, c2 = self._get_coeffs(idx, rhos=rhos)
        obj, xvals, yvals, model = bounds.solve_relu_mip(c1, c2, apx_params=apx_params)
        if not return_model:
            return obj, xvals.data, self._next_layer_relu_out(idx, xvals.data)
        else:
            return model




