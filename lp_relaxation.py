""" Kolter-Wong Linear Program estimation of robustness"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import gurobipy as gb
from collections import OrderedDict
import math
import copy
import numpy as np

import utilities as utils

from abstract_domains import Hyperbox


class LPRelax():

    def __init__(self, network, input_domain, preact_bounds=None):
        """ Sets up (but does not compute) the LinearProgramming relaxation
        ARGS:
            network: FFNet object
            input_domain : Hyperbox input domain for the network
            preact_bounds: if not None is a preactBounds object which yields ReLU input bounds
                          (if none, these will be computed on the fly)
        """

        self.network = network
        self.input_domain = input_domain
        self.var_dict = OrderedDict() # keys like "layer_num:input", or "layer_num:aux", and vals which are lists
        self.model = gb.Model()
        self.setup_done = False
        if preact_bounds is not None:
            self.input_bounds = {k: _.twocol() for k, _ in enumerate(preact_bounds)}
        else:
            self.input_bounds = {}

    @classmethod
    def get_namer(cls, layer_idx):
        base_name = '%s:input' % layer_idx
        layer_namer = lambda k: base_name + ':%s' % k
        return base_name, layer_namer

    def setup(self, lp_everything=True, verbose=True):
        """ Sets up  and computes everything """
        if self.setup_done:
            return
        self._add_vars(0, self.input_domain.twocol()) # set up input domain
        if self.input_bounds.get(0) is None:
            self.input_bounds[0] = self.input_domain.twocol()

        for idx, layer in enumerate(self.network):
            if verbose:
                print("working on layer %s : %s" % (idx, layer))
            if isinstance(layer, nn.Linear):
                self._map_linear(idx, layer)
            elif isinstance(layer, nn.Conv2d):
                self._map_conv2d(idx, layer)
            elif isinstance(layer, nn.ReLU):
                self._map_relu(idx)
            elif isinstance(layer, nn.AvgPool2d):
                self._map_avgpool(idx, layer)
            else:
                raise NotImplementedError()

            if lp_everything and idx < len(self.network) + 1:
                print("Trying to update input bounds...", idx+ 1)
                self._update_input_bounds(idx + 1)

        self.setup_done = True


    def compute(self, lp_everything=True, verbose=True):
        if not self.setup_done:
            self.setup(lp_everything=lp_everything, verbose=verbose)


        final_idx = len(self.network)
        output_var = self.var_dict[self.get_namer(final_idx)[0]][0]

        self.model.setObjective(output_var, gb.GRB.MINIMIZE)
        self.model.update()
        if verbose:
            print("Starting final optimization")
        self.model.optimize()

        return output_var, self.model


    def _add_vars(self, layer_idx, bounds_twocol, dilate=True):
        base_name, layer_namer = self.get_namer(layer_idx)
        if dilate:
            eps = 1e-8
            bounds_twocol = bounds_twocol + torch.tensor([[-eps, eps]])

        new_vars = [self.model.addVar(lb=lb, ub=ub, name=layer_namer(i))
                    for i, (lb, ub) in enumerate(bounds_twocol)]
        self.model.update()
        self.var_dict[base_name] = new_vars
        return new_vars


    def _map_linear(self, layer_idx, layer):
        # Collect input vars
        input_vars = self.var_dict[self.get_namer(layer_idx)[0]]
        input_bounds = self.input_bounds[layer_idx]

        # Set up output vars
        output_idx = layer_idx + 1
        output_box = Hyperbox(*input_bounds.T).map_linear(layer).twocol()
        self.input_bounds[output_idx] = output_box
        output_vars = self._add_vars(output_idx, output_box, dilate=False)

        # Add constraints to enforce layer action
        for output_coord, output_var in enumerate(output_vars):
            row = layer.weight[output_coord].detach().numpy()
            bias = 0.0 if (layer.bias is None) else layer.bias[output_coord]
            self.model.addConstr(output_var == gb.LinExpr(row, input_vars) + bias)
        self.model.update()


    def _map_conv2d(self, layer_idx, layer):
        input_vars = self.var_dict[self.get_namer(layer_idx)[0]]
        input_bounds = self.input_bounds[layer_idx]

        # Set up output vars
        output_idx = layer_idx + 1
        output_box = Hyperbox(*input_bounds.T).map_conv(layer).twocol()
        self.input_bounds[output_idx] = output_box
        output_vars = self._add_vars(output_idx, output_box, dilate=True)


        input_shape = layer.input_shape
        output_shape = layer(torch.zeros(input_shape).unsqueeze(0)).shape[1:]

        # Add constraints to enforce layer action:
        assert layer.dilation == (1, 1) and layer.padding_mode == 'zeros'


        input_vars = utils.conv_view(input_vars, layer.input_shape)
        output_vars = utils.conv_view(output_vars, output_shape)
        weight = layer.weight

        for out_c in range(output_shape[0]):
            for out_row in range(output_shape[1]):
                for out_col in range(output_shape[2]):
                    # For every output pixel
                    out_var = output_vars[out_c][out_row][out_col] # Variable we're setting
                    lin_expr = layer.bias[out_c].item() # Start with the bias

                    for in_c in range(input_shape[0]): # Looping over input channels
                        for ker_row in range(weight.shape[2]): # Looping over kernel rows
                            in_row = -layer.padding[0] + layer.stride[0] * out_row + ker_row
                            if not (0 <= in_row < input_shape[-2]):
                                continue  # Padding case
                            for ker_col in range(weight.shape[3]):
                                in_col = -layer.padding[1] + layer.stride[1] * out_col + ker_col
                                if not (0 <= in_col < input_shape[-1]):
                                    continue
                                weight_el = weight[out_c, in_c, ker_row, ker_col].item()
                                lin_expr += weight_el * input_vars[in_c][in_row][in_col]
                    self.model.addConstr(out_var == lin_expr)
        self.model.update()


    def _map_avgpool(self, layer_idx, layer):

        assert layer.padding == 0 # keep it simple

        # Handle inputs
        input_vars = self.var_dict[self.get_namer(layer_idx)[0]]
        input_shape = layer.input_shape
        C, H, W = input_shape[-3:]
        input_vars = utils.conv_view(input_vars, layer.input_shape)
        input_bounds = self.input_bounds[layer_idx]

        # Set up outputs
        output_idx = layer_idx + 1
        output_box = Hyperbox(*input_bounds.T).map_avgpool(layer).twocol()
        self.input_bounds[output_idx] = output_box
        output_vars = self._add_vars(output_idx, output_box, dilate=True)
        output_shape = layer(torch.zeros(input_shape).unsqueeze(0)).shape[1:]
        output_vars = utils.conv_view(output_vars, outputSha)
        # Layer specific things
        if isinstance(layer.kernel_size, int):
            kernel_size = layer.kernel_size, layer.kernel_size
        else:
            kernel_size = layer.kernel_size
        kernel_els = float(kernel_size[0] * kernel_size[1])


        output_vars = utils.conv_view(output_vars, output_shape)
        # Main loop
        for out_c in range(output_shape[0]):
            for out_row in range(output_shape[1]):
                for out_col in range(output_shape[2]):
                    out_var = output_vars[out_c][out_row][out_col]
                    lin_expr = 0.0

                    start_row = kernel_size[0] * out_row
                    start_col = kernel_size[1] * out_col
                    if ((start_row + kernel_size[0] >= H) or
                        (start_col + kernel_size[1] >= W)):
                        pass # continue # this should never happen!

                    for in_row in range(start_row, start_row + kernel_size[0]):
                        for in_col in range(start_col, start_col + kernel_size[1]):
                            lin_expr += input_vars[out_c][in_row][in_col]
                    lin_expr = lin_expr / kernel_els
                    self.model.addConstr(out_var == lin_expr)
        self.model.update()




    def _map_relu(self, layer_idx):
        input_vars = self.var_dict[self.get_namer(layer_idx)[0]]
        input_bounds = self.input_bounds[layer_idx]

        # Setup output vars
        output_idx = layer_idx + 1
        output_box = torch.relu(input_bounds) #torch.relu(input_bounds)
        self.input_bounds[output_idx] = output_box
        output_vars = self._add_vars(output_idx, output_box, dilate=False)

        # Add constraints to enforce layer action
        for coord, (input_var, output_var)  in enumerate(zip(input_vars, output_vars)):
            lb, ub = input_bounds[coord]
            if lb >= 0.0: # on case:
                self.model.addConstr(output_var == input_var)

            elif ub < 0: # off case
                self.model.addConstr(output_var == 0.0)
                #self.model.addConstr(output_var == 0.0)
            else: #unstable case
                self.model.addConstr(output_var >= 0.0)
                self.model.addConstr(output_var >= input_var)

                slope = ub / (ub - lb)
                intercept = (-lb * ub) / (ub - lb) + 1e-8 # epsilon buffer here
                self.model.addConstr(output_var <= slope * input_var + intercept)



    def _update_input_bounds(self, layer_idx, lp_everything=False):
        # Update the tightness of the bounds for layer_idx i
        vars_to_eval = self.var_dict[self.get_namer(layer_idx)[0]]
        self.model.setParam('OutputFlag', False)
        new_lbs, new_ubs = [], []
        for var in vars_to_eval:
            self.model.setObjective(var, gb.GRB.MINIMIZE)
            self.model.update()
            self.model.optimize()
            new_lb = self.model.ObjVal

            self.model.setObjective(var, gb.GRB.MAXIMIZE)
            self.model.update()
            self.model.optimize()
            new_ub = self.model.ObjVal


            var.lb = new_lb
            var.ub = new_ub
            new_lbs.append(new_lb)
            new_ubs.append(new_ub)

            self.model.update()

        new_lbs, new_ubs = torch.tensor(new_lbs), torch.tensor(new_ubs)
        self.input_bounds[layer_idx] = torch.stack([new_lbs, new_ubs],dim=1)






