""" Object to help with zonotope partitioning for dual methods """


import torch
import random
import utilities as utils
import torch.nn as nn
import math
import joblib
import numpy as np
from abstract_domains import Zonotope

class PartitionGroup():

	def __init__(self, base_zonotopes, style='fixed_dim',
				 partition_rule='random', save_partitions=True,
				 save_models=True, num_partitions=None, partition_dim=None,
				 max_order=None, force_mip=False, cache_vertices=True,
				 use_crossings=True, input_shapes=None, box_info=None,
				 num_threads=None):
		""" Parameters for partitioning.
		Two basic styles for partitioning:
			- fixed number of partitions per zonotope  (fixed_part)
				(e.g. break each zonotope into k parts)
			- fixed dimension per partition  (fixed_dim)
				(e.g. break each zonotope into subzonos of dimension k)
		ARGS:
			base_zonotopes: list of zonotopes to partition: could be None if TBD later
			partition_spec: 'fixed_dim', or 'fixed_part'
			partition_rule: 'random' or some other rule for making partitions
			save_partitions: bool - if True we save these partitions, else make every time
			save_models: bool - if True, we save the gurobi models
			num_partitions: if style is 'fixed_part', this can't be None
			partition_dim: if style is 'fixed_dim', this can't be None
			max_order: if not None, we use 'axalign' to reduce order of zonotopes
			force_mip: if True, we use MIP even in partitions of dimension 2
			cache_vertices: if True, and 2d, we save the vertices
			use_crossings: if True, we consider the coordinate-axes crossing objects in 2d zonos
			box_info: if True, we have extra boxes that we know to add in the mip
		"""
		# Assertion checks
		assert style in ['fixed_dim', 'fixed_part']
		if style == 'fixed_dim':
			assert partition_dim is not None
		else:
			assert num_partitions is not None

		self.base_zonotopes = base_zonotopes
		self.subzonos = {}
		self.groups = {}
		self.style = style
		self.partition_rule = partition_rule
		self.save_partitions = save_partitions
		self.save_models = save_models
		self.num_partitions = num_partitions
		self.partition_dim = partition_dim
		self.max_order = max_order
		self.force_mip = force_mip
		self.cache_vertices = cache_vertices
		self.use_crossings = use_crossings

		self.vertices = {}
		# Coordinate-crossings for 2d vertices
		self.crossings = {}
		self.crossing_masks = {}

		# --- box info stuff
		self.box_info = box_info
		self.vertex_masks = None
		self.box_vertices = None
		self.box_masks = None
		self.box_zono_crossings = None
		self.bxox_zono_masks = None


		self.input_shapes = input_shapes
		self.num_threads = num_threads

		if self.base_zonotopes is not None:
			self.make_all_partitions() # modifies state if save_partitions=True

	def attach_zonotopes(self, base_zonotopes):
		self.base_zonotopes = base_zonotopes
		self.make_all_partitions()

	def order_sweep(self, zono_list):
		if self.max_order is None:
			return zono_list
		outzono_list = []
		for zono in zono_list:
			if zono.order > self.max_order:
				zono = zono.reduce_simple(self.max_order, score='axalign')
			outzono_list.append(zono)
		return outzono_list

	def _access_ith_partition_dim(self, i):
		if self.partition_dim is None:
			return None
		if isinstance(self.partition_dim, int):
			return self.partition_dim
		if isinstance(self.partition_dim, dict):
			return self.partition_dim[i]

	def _access_ith_num_partitions(self, i):
		if self.num_partitions is None:
			return None
		if isinstance(self.num_partitions, int):
			return self.num_partitions
		if isinstance(self.num_partitions, dict):
			return self.num_partitions[i]


	def make_all_partitions(self, **kwargs):
		""" Generates the partitions as list of partitions,
			where each partition is a list of [(part_0_idxs), (part_1_idxs)...]"""
		return {k: self.make_ith_partition(k, **kwargs) for k in self.base_zonotopes}



	def make_ith_partition(self, i, **kwargs):
		""" Makes the partition of self.base_zonotope[i].
		"""
		zono = self.base_zonotopes[i]
		dim = zono.dim
		# First make the groups
		if self.style == 'fixed_dim':

			if isinstance(self.partition_dim, int):
				num_parts = math.ceil(dim / self.partition_dim)
			else:
				num_parts = math.ceil(dim / self.partition_dim[i])
		else:
			num_parts = self.num_partitions

		# -- can either have random or scored fxn
		if self.partition_rule == 'random':
			groups = utils.partition(list(range(zono.dim)), num_parts)

		elif self.partition_rule in ("depthwise", "spatial"):

			if i >= len(self.input_shapes) or len(self.input_shapes[i]) != 3:
				groups = utils.partition(list(range(zono.dim)), num_parts)
			else:
				shape = self.input_shapes[i]
				assert shape.numel() == zono.dim
				# Note: spatial rule should probably use both axes somehow
				axis = 0 if self.partition_rule == "depthwise" else -1
				# Note: sort of relying on Python sort being stable
				indexes = sorted(
					list(range(zono.dim)),
					key=lambda i: utils.unravel_index(i, shape)[axis]
				)
				part_dim = zono.dim // num_parts
				groups = [indexes[i: i+part_dim] for i in range(0, zono.dim, part_dim)]

		elif self.partition_rule == 'similarity':
			if self._access_ith_partition_dim(i) == 2:
				groups = self._make_similarity_groups(i)
			else:
				groups = utils.partition(list(range(zono.dim)), num_parts)
		elif self.partition_rule == 'axalign':
			if self._access_ith_partition_dim(i) == 2:
				groups = self._make_axalign_groups(i)
			else:
				groups = utils.partition(list(range(zono.dim)), num_parts)


		else:
			# if scored, then kwargs['score_fxn'] is a list of fxns (zono, idx)->score
			score_fxn = kwargs['score_fxn'][i]
			scores = torch.tensor([score_fxn(zono, j) for j in range(dim)])
			sorted_scores = list(torch.sort(dim_scores, descending=True)[1].numpy())
			groups = [sorted_scores[i: i+size] for i in range(0, dim, size)]


		# Now with groups check to see if we want to save
		if self.save_partitions:
			self.groups[i] = groups
			if self.save_models:
				subzonos = self.order_sweep([_[1] for _ in zono.partition(groups)])
				[_._setup_relu_mip2() for _ in subzonos]
				self.subzonos[i] = subzonos
			return groups
		else:
			return groups


	def _make_similarity_groups(self, idx):
		""" Makes groups based on 'similarity' of generator rows.

		Discussion:
			Idea: we do better when the zonotopes aren't square-like:
				The right answer is to compute the volume of each zono vs the box version
				(volume of a 2d zono is O(gensize^2), so not pratical to do for all O(n^2) possible 2d zonos)
				The heuristic answer is to measure similarity of dot products of abs of genrows
			Heuristic:
			- considers similarity of generator rows as the dot products of row.abs()'s
			- sorts each row according to the most similar row
			- considers highest-similarity rows first, and then creates list of pairs
		"""

		zono = self.base_zonotopes[idx]
		if zono.dim == 1:
			return [[0]]

		sim = zono.generator.abs() @ zono.generator.abs().T
		sim.fill_diagonal_(-1)

		# Each row has the index of the most sorted rows
		sort_vals, sort_idxs = torch.sort(sim, dim=1, descending=True)
		group_order = torch.sort(sort_vals[:,0], descending=True)[1]
		return self._scored_pair_groups(sort_idxs, group_order)


	def _make_axalign_groups(self, idx):
		""" Leverages the 'axalign' heuristic for order reduction to pairs

		for each candidate pair of dimensions:
			- compute score for each (2d) generator, where score is
			  l1_norm / linf_norm ,
			- get total pair-candidate score by adding up scores over all generators
		"""
		zono = self.base_zonotopes[idx]
		if zono.dim == 1:
			return [[0]]


		zono = self.base_zonotopes[idx]
		dim = zono.dim
		gen_abs = zono.generator.abs()


		# Okay, dumb algorithm loops over generators but doesn't blow up the memory
		sim = torch.zeros((zono.dim, zono.dim)).to(zono.center.device)
		counts = torch.zeros((zono.dim, zono.dim)).to(zono.center.device)
		for k in range(zono.gensize):
			# add a matrix with (i,j) element being (gen(i,k) + gen(j,k)) / max(gen(i,k), gen(j,k)) to sim
			topsum = gen_abs[:,k].view(1, -1) + gen_abs[:,k].view(-1, 1)

			botmax = torch.max(gen_abs[:,k].view(1, -1).expand(zono.dim, zono.dim),
							   gen_abs[:,k].view(-1, 1).expand(zono.dim, zono.dim))

			quotient = topsum/botmax
			quotient[quotient != quotient] = 0.0
			counts.add_((botmax > 0).int())
			sim.add_(quotient)

		sim = sim / counts
		sim[sim != sim] = 0.0
		sim.fill_diagonal_(-1)

		sort_vals, sort_idxs = torch.sort(sim, dim=1, descending=True)
		group_order = torch.sort(sort_vals[:,0], descending=True)[1]
		return self._scored_pair_groups(sort_idxs, group_order)




	def _scored_pair_groups(self, sort_idxs, group_order):
		""" Given a dxd matrix (sort idxs) where each row is a permutation of [d]
			and a group_order that's a permutation of [d], creates a list
			of pairs to maximize 'similarity'
		"""
		accounted_dims = set()
		pairs = []
		def get_pair(i):
			for el in sort_idxs[i]: # loop over this row
				el = int(el.item())
				if i == el:
					continue
				if el not in accounted_dims:
					return [i, el]


		for i in group_order:
			i = int(i.item())
			if i in accounted_dims:
				continue
			new_pair = get_pair(i)
			pairs.append(new_pair)
			accounted_dims.add(new_pair[0])
			accounted_dims.add(new_pair[1])
		return pairs


	def relu_program(self, i, c1, c2, gurobi_params=None, start=None, **kwargs):
		""" Returns (obj_val, argmin) for the i^th partition
			of relu program: min c1*z + c2*relu(z)
		ARGS:
			i: which index of partition group to handle
			c1: linear objecive vector
			c2: relu objective vector
			force_mip: don't do closed form 2d thing
			gurobi_params: dict of params to send to gurobi model
			kwargs: dict to pass to partition maker if doing on the fly
		RETURNS:
			(obj_val, argmin)
			OBJ_VAL is a LOWER BOUND, but argmin is the best primal vars
		"""
		zono = self.base_zonotopes[i]

		# Handle the unsaved partition part
		if self.groups.get(i) is None:
			groups = self.make_ith_partition(i, **kwargs)
		else:
			groups = self.groups[i]


		# ~~~~~~~~~~~~~~~~~~~~~~~New block~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		if not self.force_mip and all(len(_) == 2 for _ in groups):
			groups = utils.tensorfy(groups).long().to(zono.center.device)
			if self.vertices.get(i) is None: # if haven't computed the vertices yet...

				# Compute vertices and crossings
				vertices = zono.batch_2d_partition(groups=groups)
				crossings, crossing_masks = Zonotope.batch_2d_crossings(vertices)


				# -------------  BOX INFO STUFF
				if self.box_info is not None: # If box info exists...
					# compute box related vertices
					box = self.box_info[i]
					crossing_masks, vertex_masks = zono.batch_zono_vertex_mask(groups, vertices, box,
																			   crossings, crossing_masks)
					box_vertices, box_masks = zono.batch_box_corners_axes(groups, vertices, box)
					box_zono_crossings, box_zono_masks = zono.batch_zono_box_crossings(groups, vertices, box)
				else: # otherwise
					# Set these as None, so no errors
					vertex_masks = None
					box_vertices = box_masks = None
					box_zono_crossings = box_zono_masks = None
				# ----------------------------------------

				if self.cache_vertices: # if we want to cache....
					# Save everything
					self.vertices[i] = vertices
					self.crossings[i] = crossings
					self.crossing_masks[i] = crossing_masks
					self.vertex_masks = vertex_masks
					self.box_vertices = box_vertices
					self.box_masks = box_masks
					self.box_zono_crossings = box_zono_crossings
					self.box_zono_masks = box_zono_masks

			else:
				vertices = self.vertices[i]
				crossings = self.crossings[i]
				crossing_masks = self.crossing_masks[i]

				vertex_masks = (self.vertex_masks or {}).get(i)
				box_vertices = (self.box_vertices or {}).get(i)
				box_masks = (self.box_masks or {}).get(i)
				box_zono_crossings = (self.box_zono_crossings or {}).get(i)
				box_zono_masks = (self.box_zono_masks or {}).get(i)



			return zono.batch_2d_relu_program(c1, c2, groups=groups,
											  group_vs=vertices,
											  group_crossings=crossings,
											  group_crossing_masks=crossing_masks,
											  group_vertex_masks=vertex_masks,
											  group_box_vertices=box_vertices,
											  group_box_masks=box_masks,
											  box_zono_crossings=box_zono_crossings,
											  box_zono_masks=box_zono_masks)

		# ~~~~~~~~~~~~~~~~~~~~~~~/New block~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		# Otherwise make the partitions
		if self.subzonos.get(i) is not None:
			parts = self.subzonos[i]
		else:

			parts = self.order_sweep([_[1] for _ in self.base_zonotopes[i].partition(groups)])

		obj_val = 0.0
		argmin = torch.zeros_like(zono.center)


		#### MULTITHREADING BLOCK
		def subproblem(group, subzono):
			box_bounds = None if ((self.box_info or {}).get(i) is None) else self.box_info[i][group]
			min_val, sub_argmin, _, _ = subzono.solve_relu_mip(c1[group], c2[group], apx_params=gurobi_params,
				                                               start=start, box_bounds=box_bounds)
			return min_val, sub_argmin, group


		if self.num_threads is None:
			thread_outs = [subproblem(group, subzono) for (group, subzono) in zip(groups, parts)]
		else:
			thread_outs = joblib.Parallel(n_jobs=self.num_threads)(subproblem(group, subzono) for (group, subzono) in zip(groups, parts))
		#####

		# Resolve thread outs
		obj_val = 0
		for min_val, sub_argmin, group in thread_outs:
			obj_val += min_val
			sub_argmin = torch.tensor(sub_argmin).type(argmin.dtype).to(argmin.device).flatten()
			argmin[group] = sub_argmin
		return obj_val, argmin



	def merge_partitions(self, partition_dim=None, num_partitions=None, copy_obj=True):
		""" Merges partitions into partitions of larger dim, using the existing partitions
		ARGS:
			partition_dim/num_partitions : specs to make new partitinos
				- exactly one of these should be None
				- the other one should be eithr an int, or a dict with the same keys
				  as self.base_zonotopes
			copy_obj: bool - if True, we leave this object unchanged and compute a new object
							 otherwise, we just modify this object


		RETURNS:
			a PartitionObject
		"""
		# Merges partitions into smaller partitions using existing partitions
		# If copy is True, we create a separate partition object and leave this state

		#

		assert sum([(partition_dim is None), (num_partitions is None)]) == 1

		# Compute new groups
		new_groups = {}
		if self.style == 'fixed_dim':
			assert partition_dim is not None
			for idx, group in self.groups.items():
				current_groupsize = max(len(_) for _ in group)
				new_groupsize = self._access_ith_partition_dim(idx)

				if new_groupsize == current_groupsize:
					new_groups[idx] = group
				else:
					ratio = math.floor(new_groupsize / current_groupsize)

					new_groups[idx] = [utils.flatten(group[i: i+ratio])
									   for i in range(0, len(group), ratio)]
		if self.style == 'fixed_part':
			assert num_partitions is not None
			for idx, group in self.groups.items():
				current_numgroups = len(group)
				new_numgroups = self._access_ith_num_partitions(idx)

				if new_numgroups == current_numgroups:
					new_groups[idx] = group
				else:
					ratio = math.floor(current_numgroups / new_numgroups)
					new_groups[i] = [utils.flatten(group[i:i+ratio])
									 for i in range(0, len(group), ratio)]


		new_partition_dim = partition_dim
		new_num_partitions = num_partitions

		# Now modify the vertices
		new_vertices = {}
		for idx, group in new_groups.items():
			if group == self.groups[idx] and self.vertices.get(idx) is not None:
				new_vertices[idx] = self.vertices[idx]


		# Now make the new zonos (and setup MIPs)
		new_subzonos = {}
		if self.save_partitions:
			if self.save_models:
				for idx, group in new_groups.items():
					if max(len(_) for _ in group) <= 2: # only build MIP models if groupsize > 2
						continue
					zono = self.base_zonotopes[idx]
					if group == self.groups[idx]:
						new_subzonos[idx] = self.subzonos.get(idx) # No change to subzonos if no change to groups
						continue


					subzonos = self.order_sweep([_[1] for _ in zono.partition(group)])
					for subgroup, subzono in zip(group, subzonos):
						box_bounds = None
						if self.box_info is not None:
							box_bounds = self.box_info[idx][subgroup]
						subzono._setup_relu_mip2(box_bounds=box_bounds)
					new_subzonos[idx] = subzonos

		if copy_obj:
			new_obj = PartitionGroup(None, style=self.style,
									 partition_rule=self.partition_rule,
									 save_partitions=self.save_partitions,
									 num_partitions=num_partitions,
									 partition_dim=partition_dim,
									 max_order=self.max_order,
									 force_mip=self.force_mip,
									 cache_vertices=self.cache_vertices,
									 use_crossings=self.use_crossings,
									 box_info=self.box_info)
			new_obj.vertices = new_vertices
			new_obj.base_zonotopes = self.base_zonotopes
			new_obj.groups = new_groups
			new_obj.subzonos = new_subzonos
			return new_obj
		else:
			self.partition_dim = partition_dim
			self.num_partitions = new_num_partitions
			self.vertices = new_vertices
			self.groups = new_groups
			self.subzonos = new_subzonos
			self.box_info = box_info
			return self



	def relu_program_simplex(self, i, c1, c2):
		zono = self.base_zonotopes[i]
		if self.groups.get(i) is None:
			groups = self.make_ith_partition(i, **kwargs)
		else:
			groups = self.groups[i]

		# Otherwise make the partitions
		if self.subzonos.get(i) is not None:
			parts = self.subzonos[i]
		else:

			parts = self.order_sweep([_[1] for _ in self.base_zonotopes[i].partition(groups)])

		obj_val = 0.0
		argmin = torch.zeros_like(zono.center)
		for group, subzono in zip(groups, parts):

			min_val, argmin_ys = subzono.solve_relu_simplex(c1[group], c2[group])

			obj_val += min_val
			argmin[group] = subzono(argmin_ys)#torch.tensor(sub_argmin) # DTYPES HERE?

		return obj_val, argmin


	def relu_program_lbfgsb(self, i, c1, c2):
		zono = self.base_zonotopes[i]
		if self.groups.get(i) is None:
			groups = self.make_ith_partition(i, **kwargs)
		else:
			groups = self.groups[i]

		# Otherwise make the partitions
		if self.subzonos.get(i) is not None:
			parts = self.subzonos[i]
		else:

			parts = self.order_sweep([_[1] for _ in self.base_zonotopes[i].partition(groups)])

		obj_val = 0.0
		argmin = torch.zeros_like(zono.center)
		for group, subzono in zip(groups, parts):

			min_val, argmin_ys = subzono.solve_relu_lbfgsb(c1[group], c2[group])

			obj_val += min_val
			argmin[group] = subzono(argmin_ys)#torch.tensor(sub_argmin) # DTYPES HERE?

		return obj_val, argmin

