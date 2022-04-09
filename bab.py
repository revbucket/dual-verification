""" Preliminary Branch and Bound Schemes... """
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import time
from typing import List
import itertools
from collections import OrderedDict
from neural_nets import FFNet, PreactBounds, KWBounds, BoxInformedZonos
from abstract_domains import Hyperbox, Zonotope
import utilities as utils
from partitions import PartitionGroup
from decomposition_plnn_bounds.plnn_bounds.proxlp_solver.solver import SaddleLP
from dual_decompose import DecompDual
import experiment_utils as eu
from heapq import heappush, heappop

class Bab:
    def __init__(self, network, input_domain, split_strategy='babsr', use_intermed=True):
        self.network = network
        self.input_domain = input_domain
        self.split_strategy = split_strategy

        # Step
        ovalnet = eu.run_optprox(network, input_domain, use_intermed=use_intermed, return_model=True)[0]
        box_zonos = eu.ovalnet_to_zonoinfo(ovalnet, network, input_domain)


        self.preact_bounds = box_zonos


        # Maintain a PQ of elements like (val, split_hist) for the existing bound
        self.split_queue = []
        heappush(self.split_queue, (-float('inf'), []))


    def run_bab(self, timeout, solve_proc=None):
        if solve_proc is None:
            def solve_proc(decomp):
                decomp.manual_dual_ascent(200, verbose=50)
                return decomp.lagrangian()


        start_time = time.time()
        while time.time() - start_time < timeout and len(self.split_queue) > 0:
            val, splits = heappop(self.split_queue)
            self.solve_subproblem(val, splits, solve_proc)


        if len(self.split_queue) > 0:
            return self.split_queue[0][0]
        else:
            return True



    def solve_subproblem(self, val, splits, solve_proc):
        """Initialize a decompDual with the given splits and solve it
            Splits is a list like [(layer, coord, {+1, -1})]
        Returns:
            (val, (split_))
        """
        print("Solving subproblem: ", splits)


        # Initialize a decompDual and solve it
        split_hist = set([(_[0], _[1]) for _ in splits])
        boxzono = self.preact_bounds.from_split_hist(splits)
        decomp = DecompDual(self.network, self.input_domain, Zonotope, 'partition',
                            zero_dual=False, preact_bounds=boxzono, split_history=split_hist)
        new_val = solve_proc(decomp)


        new_val = max(val, new_val)
        if new_val >= 0:
            return

        new_split = {'babsr': decomp.find_split_babsr,
                     'fsb': decomp.find_split_fsb}[self.split_strategy]()

        pos_splits = [_ for _ in splits]
        pos_splits.append(new_split + (1,))
        neg_splits = [_ for _ in splits]
        neg_splits.append(new_split + (-1,))
        print("Pushing %s with val %s" % (splits, new_val))
        heappush(self.split_queue, (new_val, pos_splits))
        heappush(self.split_queue, (new_val, neg_splits))





