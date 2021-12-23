""" File that has neural nets with easy support for abstract interpretations
	and dual verification techniques
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from abstract_domains import Hyperbox, Zonotope, Polytope, Polytope2


class FFNet(nn.Module):
	""" Wrapper for FeedForward Neural Networks.
		Handles networks with architectures composed of sequential
		operations of:
		- Linear operators : FullyConnected
		- Nonlinear layers : ReLU
	"""

	SUPPORTED_LINS = [nn.Linear]
	SUPPORTED_NONLINS = [nn.ReLU]

	# ======================================================================
	# =           Constructor methods and helpers                          =
	# ======================================================================


	def __init__(self, net, dtype=torch.float):
		super(FFNet, self).__init__()
		self.net = net
		self.dtype = dtype
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

	def _reshape(self, x):
		if isinstance(self.net[0], nn.Linear):
			x = x.view(x.shape[0], -1)
		return x

	def forward(self, x):
		x = self._reshape(x)
		return self.net(x)


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

	def compute_polytope2(self):
		basic_bounds = PreactBounds(self.network, self.input_range, Hyperbox)
		basic_bounds.compute()
		poly2 = Polytope2(basic_bounds.bounds)
		for i, layer in enumerate(self.network, start=1):
			if isinstance(layer, nn.Linear):
				poly2.map_linear(layer, i)
			else:
				poly2.map_relu(i)

		self.bounds = poly.box_bounds
		self.polytope = poly
		self.computed = True

	def compute(self):
		if self.abstract_domain == Polytope:
			return self.compute_polytope()

		if self.abstract_domain == Polytope2:
			return self.compute_polytope2()

		if self.abstract_domain == Zonotope:
			self.input_range = self.input_range.as_zonotope()

		self.bounds.append(self.input_range)
		for layer in self.network:
			self.bounds.append(self.bounds[-1].map_layer(layer))
		self.computed = True



