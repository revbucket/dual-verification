import torch
import torch.nn as nn
import torch.nn.functional as F
import gurobipy as gb
from collections import OrderedDict
import random
import utilities as utils


class AbstractDomain:
	""" Abstract class that handles forward passes """
	def map_layer(self, layer):
		if isinstance(layer, nn.Linear):
			return self.map_linear(layer)
		elif isinstance(layer, nn.ReLU):
			return self.map_relu()



# ==================================================================
# =                     HYPERBOXES                                 =
# ==================================================================


class Hyperbox(AbstractDomain):

	def __init__(self, lbs, ubs):
		self.lbs = lbs
		self.ubs = ubs
		self.center = (self.lbs + self.ubs) / 2.0
		self.rad = self.ubs - self.center

	def cuda(self):
		self.lbs = self.lbs.cuda()
		self.ubs = self.ubs.cuda()

	def cpu(self):
		self.lbs = self.lbs.cpu()
		self.ubs = self.ubs.cpu()

	def twocol(self):
		return torch.stack([self.lbs, self.ubs])

	def dim(self):
		return self.lbs.numel()

	@classmethod
	def linf_box(cls, center, rad):
		dim = center.numel()
		if isinstance(rad, float):
			rad = torch.ones_like(center) * rad
		return cls(center - rad, center + rad)

	@classmethod
	def from_zonotope(cls, zonotope):
		center = zonotope.center
		rad = zonotope.generator.abs().sum(dim=1)
		return cls.linf_box(center, rad)

	def as_zonotope(self):
		return Zonotope.from_hyperbox(self)

	def contains(self, x):
		return ((self.lbs <= x) * (x <= self.ubs)).min().item()

	def clamp(self, lo, hi):
		if isinstance(lo, (float, int)):
			lo = torch.ones_like(self.lbs) * lo
		if isinstance(hi, (float, int)):
			hi = torch.ones_like(self.ubs) * hi

		new_lbs = torch.stack([lo, self.lbs]).max(dim=0)[0]
		new_ubs = torch.stack([hi, self.ubs]).min(dim=0)[0]
		return Hyperbox(new_lbs, new_ubs)


	def solve_lp(self, obj: torch.Tensor, get_argmin: bool = False):
		""" Solves linear programs like min <obj, x> over hyperbox
		ARGS: obj : objective vector
			  get_opt_point : if True, returns optimal point too
		RETURNS: either optimal_value or (optimal_value, optimal point)
		"""
		# Do this via centers and rads
		signs = torch.sign(obj)
		opt_point = F.relu(signs) * self.lbs + F.relu(-signs) * self.ubs
		opt_val = obj @ opt_point
		if get_argmin:
			return opt_val, opt_point
		return opt_val


	# =============================================
	# =           Pushforward Operators           =
	# =============================================

	def map_linear(self, linear):
		# Maps Hyperbox through linear layer
		new_center = linear(self.center)
		new_rad = linear.weight.abs() @ self.rad
		return Hyperbox.linf_box(new_center, new_rad)

	def map_relu(self):
		# Maps Hyperbox through ReLU layer
		return Hyperbox(F.relu(self.lbs), F.relu(self.ubs))


	# ======  End of Pushforward Operators  =======


# ========================================================
# =                      ZONOTOPES                       =
# ========================================================




class Zonotope(AbstractDomain):

	def __init__(self, center, generator):
		self.center = center
		self.generator = generator
		hbox = self.as_hyperbox()
		self.lbs = hbox.lbs
		self.ubs = hbox.ubs
		self.rad = (self.ubs - self.lbs) / 2.0

		self.dim = self.lbs.numel()
		self.order = self.generator.shape[1] / self.dim

	def __call__(self, y):
		return self.center + self.generator @ y

	def cuda(self):
		self.center = center.cuda()
		self.generator = generator.cuda()

	def cpu(self):
		self.center = center.cpu()
		self.generator = generator.cpu()

	@classmethod
	def from_hyperbox(cls, hyperbox):
		center = hyperbox.center
		generator = torch.diag(hyperbox.rad)
		return cls(center, generator)

	def as_hyperbox(self):
		return Hyperbox.from_zonotope(self)

	def solve_lp(self, obj: torch.Tensor, get_argmin: bool = False):
		# RETURNS: either optimal_value or (optimal_value, optimal point)
		center_val = self.center @ obj
		opt_signs = -torch.sign(self.generator.T @ obj)
		opt_point = self.center + self.generator @ opt_signs
		opt_val = opt_point @ obj
		if get_argmin:
			return (opt_val, opt_point)
		return opt_val

	def collect_vertices(self, num_vs):
		""" Collects num_vs random vertices (not sampled uniformly)
		"""
		vs =[]
		for i in range(num_vs):
			obj = torch.randn_like(self.center)
			vs.append(self.solve_lp(obj, True)[1])
		return vs

	def contains(self, x):
		""" Returns True iff x is in the zonotope """
		return self.y(x) is not None


	def y(self, x):
		""" Takes in a point and either returns the y-val
			such that c+Ey = x
		~or~ returns None (if no such y exists)
		"""
		model = gb.Model()

		yvars = model.addVars(range(self.generator.shape[1]), lb=-1.0, ub=1.0)
		yvars = [yvars[_] for _ in range(self.generator.shape[1])]
		model.update()
		for i in range(self.generator.shape[0]):
			print(self.generator[i].shape, len(yvars))
			model.addConstr(x[i].item() == self.center[i].item() +
							gb.LinExpr(self.generator[i], yvars))


		model.setObjective(0.0)
		model.update()
		model.optimize()

		if model.Status == 2:
			return np.array([y.x for y in yvars])
		return None


	def partition(self, groups):
		""" Partitions this input into multiple zonotopes of different coordinates
			e.g. Just groups the zonotopes into multiple zonotopes based on coordinate
				 indices
		ARGS:
			- groups is a list of index-lists. Final one may be omitted/inferred
		"""
		groups = utils.complete_partition(groups, self.lbs.numel())

		out_zonos = []
		for group in groups:
			out_zonos.append(Zonotope(self.center[group], self.generator[group]))
		return out_zonos


	def reduce_simple(self, order, score='norm'):
		""" Does the simplest order reduction possible
		Keeps the (order-1) * dim largest 2-norm generator columns
		And then adds a minkowski sum of the remainder to split the differenc

		score:
			'norm': keeps longest generators
			'axalign': scores based on  ||g||_1 - ||g||_infty
		"""

		if score == 'norm':
			scores = self.generator.norm(dim=0, p=2)
		else:
			scores = self.generator.norm(dim=0, p=1) -\
					 self.generator.norm(dim=0, p=float('inf'))
		sorted_idxs = torch.sort(scores, descending=True).indices
		keep_num = int((order - 1) * self.lbs.numel())
		keep_idxs = sorted_idxs[:keep_num]
		keep_gens = self.generator[:, keep_idxs]

		trash_idxs = sorted_idxs[keep_num:]
		trash_gens = self.generator[:, trash_idxs]
		box_gens = torch.diag(trash_gens.abs().sum(dim=1))

		return Zonotope(self.center, torch.cat([keep_gens, box_gens], dim=1))



	# =============================================
	# =           Pushforward Operators           =
	# =============================================
	def map_linear(self, linear):
		new_center = linear(self.center)
		new_generator = linear.weight @ self.generator
		return Zonotope(new_center, new_generator)


	def map_relu(self):
		""" Remember how to do this...
			Want to minimize the vertical deviation on
			|lambda * (c + Ey) - ReLU(c+Ey)| across c+Ey in [l, u]
		"""
		#### SOME SORT OF BUG HERE --- NOT PASSING SANITY CHECKS
		new_center = torch.clone(self.center)
		new_generator = torch.clone(self.generator)

		on_neurons = self.lbs > 0
		off_neurons = self.ubs < 0
		unstable = ~(on_neurons + off_neurons)
		# For all 'on' neurons, nothing needs doing
		# For all 'off' neurons, set to zero
		new_center[off_neurons] = 0
		new_generator[off_neurons, :] = 0

		# Recipe for unstable neurons:
		# 1) multiply current generator by u/u-l
		# 2) multiple current center by u/u-l
		# 3) add -ul/2*(u-l) to current center
		# 4) add new column vec to generator with -ul/2*(u-l) to matrix
		scale = self.ubs[unstable] / (self.ubs[unstable] - self.lbs[unstable])
		offset = -self.lbs[unstable] * scale / 2.0

		new_generator[unstable] *= scale.view(-1, 1) # 1
		new_center[unstable] *= scale                # 2
		new_center[unstable] += offset               # 3

		new_cols = torch.zeros_like(self.lbs.view(-1, 1).expand(-1, unstable.sum())) # 4
		new_cols[unstable] = torch.diag(offset)
		return Zonotope(new_center, torch.cat([new_generator, new_cols], dim=1))

	# ======  End of Pushforward Operators  =======


	# ===============================================================
	# =           MIP-y things                                      =
	# ===============================================================

	def _encode_mip(self):
		""" Creates a gurobi model for the zonotope
		Variables are the y-variables and equality constraints for the xs

		RETURNS: dict
			{model: Model object
			 xs: x variables
			 ys: y variables }
		"""
		model = gb.Model()
		y_namer = namer('y')
		ys = [model.addVar(lb=-1, ub=1, name=y_namer(i))
		 	  for i in range(self.generator.shape[1])]
		model.update()

		x_namer = namer('x')
		xs = []
		eps = 1e-6
		for idx, gen_row in enumerate(self.generator):
			xs.append(model.addVar(lb=self.lbs[idx] - eps,
			         			   ub=self.ubs[idx] + eps,
				         		   name=x_namer(idx)))
			model.addConstr(xs[-1] == gb.LinExpr(gen_row, ys) +
										 self.center[idx])
		model.update()

		return {'model': model,
		        'xs': xs,
		        'ys': ys}



	def solve_relu_mip(self, c1, c2, apx_params=None, verbose=False):
		""" Solves the optimization program:
			min c1*z + c2*Relu(z) over this zonotope

		if apx_params is None, we return only a LOWER BOUND
		on the objective value
		Returns
			(opt_val, argmin x, argmin y)
		"""
		mip_dict = self._encode_mip()
		model = mip_dict['model']

		for k,v in (apx_params or {}).items():
			model.setParam(k, v)

		if verbose is False:
			model.setParam('OutputFlag', False)


		xs, ys = mip_dict['xs'], mip_dict['ys']

		# Now add ReLU constraints, using the big-M encoding
		unc_idxs = ((self.lbs * self.ubs) < 0).nonzero().squeeze()
		zs = []
		z_namer = namer('z') # integer variables
		relu_namer = namer('relu')
		relu_vars = {}
		for idx in unc_idxs:
			idx = idx.item()
			lb, ub = self.lbs[idx], self.ubs[idx]
			relu_vars[idx] = model.addVar(lb=0, ub=ub, name=relu_namer(idx))
			zs.append(model.addVar(vtype=gb.GRB.BINARY, name=z_namer(idx)))
			model.addConstr(relu_vars[idx] >= xs[idx])
			model.addConstr(relu_vars[idx] <= zs[-1] * ub)
			model.addConstr(relu_vars[idx] <= xs[idx] - (1- zs[-1]) * lb)
		model.update()

		# And then add objectives
		zero_var = model.addVar(lb=0, ub=0, name='zero')
		all_relu_vars = []
		for idx in range(self.generator.shape[0]):
			if self.lbs[idx] >= 0:
				all_relu_vars.append(xs[idx])
			elif self.ubs[idx] <= 0:
				all_relu_vars.append(zero_var)
			else:
				all_relu_vars.append(relu_vars[idx])

		model.setObjective(gb.LinExpr(c1, xs) + gb.LinExpr(c2, all_relu_vars),
						   gb.GRB.MINIMIZE)
		model.update()

		# And solve and read solutions
		model.optimize()

		if apx_params is not None:
			return model.ObjBound, model
		obj = model.objVal
		xvals = [_.x for _ in xs]
		yvals = [_.x for _ in ys]

		return obj, xvals, yvals, model


	def k_group_relu(self, c1, c2, k=10):
		# Randomly partitions into groups of size k and solves relu programming over prismed zonos
		dim = self.lbs.numel()
		gap = dim // k
		idxs = list(range(dim))
		random.shuffle(idxs)
		groups = [idxs[i::gap] for i in range(gap)]
		opt_point = torch.zeros_like(self.center)
		zonos = self.partition(groups)

		outputs = []
		for z, g in zip(zonos, groups):
			this_out = z.solve_relu_mip(c1[g], c2[g])
			outputs.append(this_out[0])
			opt_point[g] = torch.tensor(this_out[1])

		return sum(outputs), opt_point


# ========================================================
# =                      POLYTOPES                       =
# ========================================================

def namer(pfx_str):
	# Get prefix namer
	return lambda idx: pfx_str + ':' + str(idx)

def relu(float_val):
	return max([float_val, 0.0])

class Polytope(AbstractDomain):
	def __init__(self, box_constraint, lp_everything=False):
		""" Gurobi variables named like:
		x_i:j --> the j'th neuron of the i'th X layer (post-ReLU)
		z_i:j --> the j'th neuron of the i'th Z layer (pre-ReLU)
		and are stored layerwise in the var_dict with keys like
		'x_i', 'z_j'... which points to a LIST of the vars

		ARGS:
			box_constraint: is a hyperbox constraining the inputs
		"""
		self.var_dict = OrderedDict()
		self.model = gb.Model()
		self.model.setParam('OutputFlag', False)
		self.bounds = OrderedDict() # box bounds for each variable here
		self.num_relus = 0
		self.lp_everything = lp_everything

		# First create the input variables
		num_init_vars = box_constraint.lbs.numel()
		layer_name = 'x1'
		pfx = namer(layer_name)
		this_layer = []
		for idx in range(num_init_vars):
			name = pfx(idx)
			this_layer.append(self.model.addVar(lb=box_constraint.lbs[idx],
							       				ub=box_constraint.ubs[idx],
												name=name))

		self.model.update()
		self.var_dict[layer_name] = this_layer
		self.bounds[layer_name] = box_constraint




	def map_linear(self, linear):
		""" Maps this object through a linear layer """
		last_layer = 'x' + str(self.num_relus + 1)
		last_vars = self.var_dict[last_layer]
		last_bounds = self.bounds[last_layer]


		# First create new bounds
		layer_name = 'z' + str(self.num_relus + 1)
		pfx = namer(layer_name)
		new_bounds = last_bounds.map_linear(linear)
		self.bounds[layer_name] = new_bounds

		# Then create new variables
		this_layer = []
		for idx in range(linear.out_features):
			name = pfx(idx)
			this_layer.append(self.model.addVar(lb=new_bounds.lbs[idx],
				                                ub=new_bounds.ubs[idx],
				                                name=name))
		self.model.update()
		self.var_dict[layer_name] = this_layer

		# Then add the constraints
		for idx in range(linear.out_features):
			rowvec = linear.weight[idx].data.cpu().numpy()
			bias = linear.bias[idx].data
			self.model.addConstr(gb.LinExpr(rowvec, last_vars) + bias == this_layer[idx])

		self.model.update()

		if self.lp_everything:
			self.bounds[layer_name] = self.get_tighter_layer_bounds(layer_name)

	def map_relu(self):
		""" Maps this object through a ReLU layer: uses the triangle relaxation
		"""
		eps = 1e-6
		last_layer = 'z' + str(self.num_relus + 1)
		last_vars = self.var_dict[last_layer]
		last_bounds = self.bounds[last_layer]

		# First create the new bounds
		layer_name = 'x' + str(self.num_relus + 2)
		pfx = namer(layer_name)
		new_bounds = last_bounds.map_relu()
		self.bounds[layer_name] = new_bounds

		# Then create new variables
		this_layer = []
		for idx in range(len(last_vars)):
			name = pfx(idx)
			this_layer.append(self.model.addVar(lb=relu(new_bounds.lbs[idx]),
				 							    ub=relu(new_bounds.ubs[idx]),
				 							    name=name))
		self.model.update()
		self.var_dict[layer_name] = this_layer

		# And add triangle constraints...
		for idx in range(len(last_vars)):
			l,u = last_bounds.lbs[idx], last_bounds.ubs[idx]
			input_var = last_vars[idx]
			output_var = this_layer[idx]
			if l > 0: # Relu always ON
				self.model.addConstr(input_var == output_var)

			elif u < 0: #ReLU always OFF
				self.model.addConstr(output_var == 0.0)
			else:
				continue
				self.model.addConstr(output_var >= 0.0)
				self.model.addConstr(output_var >= input_var)
				slope = u / (u - l)
				intercept = l * slope
				self.model.addConstr(output_var <= gb.LinExpr(slope, input_var) - intercept)
			self.model.update()

		self.num_relus += 1
		if self.lp_everything:
			self.bounds[layer_name] = self.get_tighter_layer_bounds(layer_name)


	def get_tighter_layer_bounds(self, layer_str):
		this_layer = self.var_dict[layer_str]

		lbs, ubs = [], []
		for var in this_layer:
			self.model.setObjective(var, gb.GRB.MINIMIZE)
			self.model.update()
			self.model.optimize()
			lbs.append(self.model.ObjVal)

			self.model.setObjective(var, gb.GRB.MAXIMIZE)
			self.model.update()
			self.model.optimize()
			ubs.append(self.model.ObjVal)

		return Hyperbox(torch.Tensor(lbs), torch.Tensor(ubs))








