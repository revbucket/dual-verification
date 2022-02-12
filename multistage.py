""" Handles the multistage setting """

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from collections import OrderedDict
from neural_nets import FFNet, PreactBounds, KWBounds, BoxInformedZonos
from abstract_domains import Hyperbox, Zonotope
import utilities as utils
from partitions import PartitionGroup
from dual_decompose import DecompDual


class MultiStageDual:


    def __init__(self, network, input_domain, preact_domain=Zonotope,
                 choice='partition', final_dims=None,
                 opt_method='adam', opt_params=None):
        """ Sets up to run the multistage dual optimization.
            Loops over every prefix net and uses this to compute a box-informed zonotope
            (which ultimately computes final bounds)
        ARGS:
            network: FFNet object we are verifying
            input_domain: Hyperbox input range
            preact_domain: Abstract domain we propagate throughout
            choice: optimization choice -- partition by default means 2d-partitions
            final_dims: if not None, is a dict with indices for what we want the final
                        layer partition dim to be
            adam_params: if not none, has some extra info for how we want to perform adam (manually)
        """

        self.network = network
        self.input_domain = input_domain
        self.preact_domain = preact_domain
        self.choice = choice
        self.final_dims = final_dims


        self.opt_method = opt_method
        self.opt_params = None




    def compute(self, num_iters, verbose=False):
        """ Does the whole multistage setting... extra kwargs to be added for fine-grained control
        ARGS:
            num_iters: (int) number of iterations per layer
            verbose: same verbose-rule as in DecompDual
        """
        # First compute the indexes we need
        all_idxs = [i +1 for (i, layer) in enumerate(self.network)
                    if isinstance(layer, (nn.Linear, nn.Conv2d))][1:]


        # Establish first bounds...
        box_range = [None, self.input_domain.map_layer(self.network[0])]

        # ... and then loop through all others
        for idx in all_idxs:
            start_time = time.time()
            prefix_net = self.network.prefix(idx)
            preact_bounds = BoxInformedZonos(prefix_net, self.input_domain,
                                             box_range=box_range).compute()

            # Run the dual ascent on this layer to get a tighter box bound
            new_box, decomp_obj = self.compute_ith_layer(idx, preact_bounds,
                                                         num_iters, verbose=verbose)
            if verbose != False:
                print("Finished layer %s in %.02f" % (idx, time.time() - start_time))
            box_range.append(new_box)

        return box_range, decomp_obj


    def compute_ith_layer(self, idx, preact_bounds, num_iters, verbose=False):
        """ Uses DecompDual to get a new box bound for the output layer"""
        prefix_net = self.network.prefix(idx)
        decomp = DecompDual(prefix_net, self.input_domain,
                            preact_domain=self.preact_domain,
                            choice='naive',
                            preact_bounds=preact_bounds,
                            compute_all_bounds=True)
        decomp.manual_dual_ascent(num_iters, verbose=verbose,
                                  optim_params=self.opt_params)
        decomp.choice = 'partition'
        decomp.manual_dual_ascent(200, verbose=verbose,
                                  optim_params=self.opt_params)
        if self.final_dims is not None:
            # Maybe a key here?
            argkey = max(decomp.partition.base_zonotopes) - 2
            decomp.merge_partitions(partition_dim={argkey: self.final_dim[argkey]})

        lbubs = decomp.get_lbub_bounds()
        new_box = Hyperbox(lbubs[:,0], lbubs[:,1])

        return new_box, decomp

