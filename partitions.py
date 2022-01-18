""" Object to help with zonotope partitioning for dual methods """


import torch
import random
import utilities as utils
import torch.nn as nn
import math

class PartitionGroup():

	def __init__(self, base_zonotopes, style='fixed_dim',
				 partition_rule='random', save_partitions=True,
				 save_models=True, num_partitions=None, partition_dim=None,
				 max_order=None, force_mip=False, cache_vertices=True,
				 use_crossings=True):
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
			num_parts = math.ceil(dim / self.partition_dim)
		else:
			num_parts = self.num_partitions

		# -- can either have random or scored fxn
		if self.partition_rule == 'random':
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


		# Now handle the twoD case if we can
		if not self.force_mip and all(len(_) == 2 for _ in groups):
			# group_vs = zono.batch_2d_partition(groups=groups)
			# return zono.batch_2d_relu_program(c1, c2, groups=torch.tensor(groups))#, group_vs=group_vs)
			groups = utils.tensorfy(groups).long().to(zono.center.device)
			if not self.cache_vertices:
				if self.use_crossings:
					return zono.loop_zono_rp(c1, c2, groups=groups) # uses crossings by default
				else:
					return zono.batch_2d_relu_program(c1, c2, groups=groups)
			else:
				if self.vertices.get(i) is None:
					if self.use_crossings:
						self.vertices[i] = zono.loop_batch_zono_vs(groups=groups)
					else:
						self.vertices[i] = zono.batch_2d_partition(groups=groups)
				#if self.use_crossings:
				#	return zono.loop_zono_rp(c1, c2, groups=groups, loop_outs=self.vertices[i])
				#else:
				return zono.batch_2d_relu_program(c1, c2, groups=groups,
													  group_vs=self.vertices[i])


		# Otherwise make the partitions
		if self.subzonos.get(i) is not None:
			parts = self.subzonos[i]
		else:

			parts = self.order_sweep([_[1] for _ in self.base_zonotopes[i].partition(groups)])

		obj_val = 0.0
		argmin = torch.zeros_like(zono.center)
		for group, subzono in zip(groups, parts):
			min_val, sub_argmin, _, _ = subzono.solve_relu_mip(c1[group], c2[group], apx_params=gurobi_params,
				                                               start=start)
			obj_val += min_val

			sub_argmin = torch.tensor(sub_argmin).type(argmin.dtype).to(argmin.device).flatten()
			argmin[group] = sub_argmin

		return obj_val, argmin


	def merge_partitions(self, partition_dim=None, num_partitions=None, copy_obj=True):
		""" Merges partitions into partitions of larger dim, using the existing partitions
		ARGS:
			partition_dim/num_partitions : specs to make new partitinos
			copy_obj: bool - if True, we leave this object unchanged and compute a new object
							 otherwise, we just modify this object
		RETURNS:
			a PartitionObject
		"""
		# Merges partitions into smaller partitions using existing partitions
		# If copy is True, we create a separate partition object and leave this state


		if self.style == 'fixed_dim':
			assert partition_dim is not None
			ratio = math.floor(partition_dim / self.partition_dim)
		elif self.style == 'fixed_part':
			assert num_partitions is not None
			ratio = math.floor(num_partitions / self.num_partitions)
		else:
			raise NotImplementedError()
		new_partition_dim = partition_dim
		new_num_partitions = num_partitions

		# No need to merge if not saving groups anyway
		new_vertices = {} # uncache vertices, regardless
		if self.save_partitions is False:
			return {}

		# Figure out how to merge groups:...
		new_groups = {}
		for i, group in self.groups.items():
			new_groups[i] = [utils.flatten(group[i:i+ratio])
						  	 for i in range(0, len(group), ratio)]

		new_subzonos = {}
		if self.save_partitions:
			if self.save_models:
				for i, group in new_groups.items():
					zono = self.base_zonotopes[i]
					subzonos = self.order_sweep([_[1] for _ in zono.partition(group)])
					[_._setup_relu_mip2() for _ in subzonos]
					new_subzonos[i] = subzonos

		if copy_obj:
			new_obj = PartitionGroup(None, style=self.style,
									 partition_rule=self.partition_rule,
									 save_partitions=self.save_partitions,
									 save_models=self.save_models,
									 num_partitions=new_num_partitions,
									 partition_dim=new_partition_dim,
							         max_order=self.max_order,
							         force_mip=self.force_mip,
							         cache_vertices=self.cache_vertices,
							         use_crossings=self.use_crossings,
							         )
			new_obj.vertices = new_vertices
			new_obj.base_zonotopes = self.base_zonotopes
			new_obj.groups = new_groups
			new_obj.subzonos = new_subzonos
			return new_obj
		else:
			self.partition_dim = new_partition_dim
			self.num_partitions = new_num_partitions
			self.vertices = new_vertices
			self.groups = new_groups
			self.subzonos = new_subzonos
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
