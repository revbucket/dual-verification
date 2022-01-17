""" Easy MIP verification for true robustness """

import torch
import torch.nn as nn
import torch.nn.functional as F
import gurobipy as gb
from collections import OrderedDict
import math
import copy
import numpy as np
import utilities as utils
from abstract_domains import Hyperbox, Zonotope
from neural_nets import PreactBounds, KWBounds
from lp_relaxation import LPRelax


class MIPVerify(LPRelax):

    def __init__(self, network, input_domain, preact_bounds):
        """ Sets up the MIP verification model (but does not compute)
        ARGS:
            network: FFNet object
            input_domain: Hyperbox input domain for the network
            preact_bounds : if not None, is a preactBonuds object which yields
                            ReLU input bounds
        """
        assert preact_bounds is not None
        LPRelax.__init__(self, network, input_domain, preact_bounds)
        # Has attrs {network, input_domain, var_dict, model, setup_done, input_bounds}


    def setup(self, verbose=True):
        if self.setup_done:
            return
        self._add_vars(0, self.input_domain.twocol())

        for idx, layer in enumerate(self.network):
            if verbose:
                print("Working on layer %s: %s" % (idx, layer))
            if isinstance(layer, nn.Linear):
                self._map_linear(idx, layer)
            elif isinstance(layer, nn.Conv2d):
                self._map_conv2d(idx, layer)
            elif isinstance(layer, nn.ReLU):
                self._map_relu(idx)
            else:
                raise NotImplementedError()

        self.setup_done = True

    def compute(self, verbose=True, gurobi_params=None):
        # Returns (lb, ub, model)
        if not self.setup_done:
            self.setup(verbose)

        if gurobi_params is None:
            gurobi_params = {}

        gurobi_params['OutputFlag'] = verbose
        for k,v in gurobi_params.items():
            self.model.setParam(k, v)
        self.model.update()

        final_idx = len(self.network)
        output_var = self.var_dict[self.get_namer(final_idx)[0]][0]
        self.model.setObjective(output_var, gb.GRB.MINIMIZE)

        self.model.update()
        self.model.optimize()

        bound = self.model.ObjBound
        val = self.model.ObjVal

        return bound, val, self.model

    def _map_relu(self, layer_idx):
        tolerance = 0#1e-8
        input_vars = self.var_dict[self.get_namer(layer_idx)[0]]
        relu_name, relu_namer = self.get_namer(layer_idx, relu=True)
        input_bounds = self.input_bounds[layer_idx]

        output_idx = layer_idx + 1
        output_box = torch.relu(input_bounds)

        output_vars = self._add_vars(output_idx, output_box, dilate=False)

        relu_vars = {}
        for coord, (input_var, output_var) in enumerate(zip(input_vars, output_vars)):
            lb, ub = input_bounds[coord]
            if lb >= 0:
                self.model.addConstr(output_var == input_var)
            elif ub < 0:
                self.model.addConstr(output_var == 0.0)
            else:
                # Do big M thing here...
                relu_var = self.model.addVar(lb=0, ub=1, name=relu_namer(coord),
                                             vtype=gb.GRB.BINARY)
                relu_vars[coord] = relu_var


                self.model.addConstr(output_var >= input_var -tolerance)
                self.model.addConstr(output_var >= 0 -tolerance)
                self.model.addConstr(output_var <= ub * relu_var + tolerance)
                self.model.addConstr(output_var <= input_var - lb * (1- relu_var) + tolerance)
        self.var_dict[relu_name] = relu_vars
        self.model.update()

