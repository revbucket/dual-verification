import torch
import torch.nn as nn
import torch.nn.functional as F
import gurobipy as gb
from collections import OrderedDict
import random
import utilities as utils
import itertools
import math
import copy
import numpy as np

class AbstractDomain:
    """ Abstract class that handles forward passes """
    def map_layer(self, layer):
        if isinstance(layer, nn.Linear):
            return self.map_linear(layer)
        elif isinstance(layer, nn.Conv2d):
            return self.map_conv(layer)
        elif isinstance(layer, nn.ReLU):
            return self.map_relu()
        elif isinstance(layer, nn.AvgPool2d):
            return self.map_avgpool(layer)



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
        return torch.stack([self.lbs, self.ubs]).T

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

    def map_conv(self, conv):
        # Maps hyperbox through convolutional layer


        new_center = conv(self.center.view(-1, *conv.input_shape))
        new_rad = conv._conv_forward(self.rad.view(-1, *conv.input_shape),
                                     conv.weight.abs(), None)
        return Hyperbox.linf_box(torch.flatten(new_center),
                                 torch.flatten(new_rad))

    def map_relu(self):
        # Maps Hyperbox through ReLU layer
        return Hyperbox(F.relu(self.lbs), F.relu(self.ubs))

    def map_avgpool(self, layer):
        new_center = layer(self.center.view(-1, *layer.input_shape))
        new_rad = layer(self.rad.view(-1, *layer.input_shape))

        return Hyperbox.linf_box(torch.flatten(new_center),
                                 torch.flatten(new_rad))



    # ======  End of Pushforward Operators  =======


# ========================================================
# =                      ZONOTOPES                       =
# ========================================================




class Zonotope(AbstractDomain):

    def __init__(self, center, generator, max_order=100):
        self.center = center
        self.generator = generator
        self.keep_mip = True
        self.relu_prog_model = None
        self.max_order = max_order

        self._compute_state()


    def _compute_state(self):
        hbox = self.as_hyperbox()
        self.lbs = hbox.lbs
        self.ubs = hbox.ubs
        self.rad = (self.ubs - self.lbs) / 2.0
        self.dim = self.lbs.numel()
        self.order = self.generator.shape[1] / self.dim


    def __call__(self, y):
        return self.center + self.generator @ y

    @classmethod
    def merge_partitions(cls, partitions, num_groups=1):
        """ Merges partitions into a single zonotope
        ARGS:
            partitions: list of ([idx...], zono) pairs
            num_groups: how many
        RETURNS:
            list of ([idx...], zono) pairs of length num_groups
        """
        if num_groups > 1:
            idx_groups = [list(range(k, len(partitions), num_groups)) for k in range(num_groups)]
        else:
            idx_groups = [list(range(len(partitions)))]

        outputs = []
        for idx_group in idx_groups:
            idxs = list(itertools.chain(*[partitions[i][0] for i in idx_group]))
            center = torch.cat([partitions[i][1].center for i in idx_group])
            generator = torch.cat([partitions[i][1].generator for i in idx_group])
            outputs.append((idxs, cls(center, generator)))

        return outputs




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

    def as_zonotope(self):
        return self

    def twocol(self):
        return torch.stack([self.lbs, self.ubs]).T

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

    def contains(self, x, silent=True):
        """ Returns True iff x is in the zonotope """
        return self.y(x, silent=silent) is not None


    def y(self, x, silent=True):
        """ Takes in a point and either returns the y-val
            such that c+Ey = x
        ~or~ returns None (if no such y exists)
        """
        model = gb.Model()
        if silent:
            model.setParam('OutputFlag', False)
        eps = 1e-6
        yvars = model.addVars(range(self.generator.shape[1]), lb=-1.0-eps, ub=1.0+eps)
        yvars = [yvars[_] for _ in range(self.generator.shape[1])]
        model.update()
        for i in range(self.generator.shape[0]):
            model.addConstr(x[i].item() >= self.center[i].item() +
                            gb.LinExpr(self.generator[i], yvars) - eps)
            model.addConstr(x[i].item() <= self.center[i].item() +
                            gb.LinExpr(self.generator[i], yvars) + eps)

        model.setObjective(0.0)
        model.update()
        model.optimize()

        if model.Status == 2:
            return torch.tensor([y.x for y in yvars])
        return None

    def split_on_coord(self, idx):
        """ Creates two zonotopes splitting this along a single coordinate
            (for branch and bound?)
        """
        assert self.lbs[idx] * self.ubs[idx] < 0 # only split where possible

        # hacking HARD here:
        SHIFT = 100
        pos_shift = self.lbs - SHIFT
        pos_shift[idx] = 0.0

        pos_center = torch.clone(self.center) - pos_shift
        pos_zono = Zonotope(pos_center, self.generator).map_relu()

        pos_zono.center += pos_shift
        pos_zono._compute_state()


        def reflect_zono(zono, idx=idx):
            zono.center[idx] *= -1
            zono.generator[idx] *= -1
            zono._compute_state()

        neg_shift = -self.ubs - SHIFT
        neg_shift[idx] = 0.0
        refl_neg_zono = Zonotope(-self.center - neg_shift, -self.generator).map_relu()
        refl_neg_zono.center += neg_shift
        refl_neg_zono.center *= -1
        refl_neg_zono.generator *= -1
        refl_neg_zono._compute_state()

        return pos_zono, refl_neg_zono

        clone_zono.center -= neg_shift
        out_zono = Zonotpe
        reflect_zono
        negate_coord = torch.linear(self.dim, self.dim)




        output_zonos = []
        for negate in [True, False]:
            center = torch.clone(self.center)
            lb, ub = self.lbs[idx].item(), self.ubs[idx].item()
            if negate:
                for to_negate in [center[idx], lb, ub]:
                    to_negate *= -1
                lb, ub = ub, lb
            scale = ub / (ub - lb)
            offset = -lb * scale / 2.0

            center[idx] *= scale
            center += offset


            new_col = torch.zeros_like(self.lbs.view(-1, 1))
            new_col[idx] = offset
            generator = torch.cat([self.generator, new_col], dim=1)
            generator[idx][:-1] *= scale
            if negate:
                center[idx] *= -1

            output_zonos.append(Zonotope(center, generator, max_order=self.max_order))

        return output_zonos


    def fw(self, f, num_iter=10000, rand_init=True):
        # f is a function that operates on x in Z
        # Runs frank wolfe on this zonotope

        f_prime = lambda y: f(self(y))
        # Setup solvers
        def get_grad(y): # finds direction of x that minimizes f
            x = self(y).detach().requires_grad_()
            f(x).backward()
            return x.grad

        def s_plus(y): # returns y that minimizes linear apx of f'
            return -torch.sign(get_grad(y) @ self.generator)

        if rand_init:
            y = torch.rand_like(self.generator[0]) * 2 - 1.0
        else:
            y = torch.zeros_like(self.generator[0])
        y = y.detach().requires_grad_()

        for step in range(num_iter):
            gamma = 2 / float(step + 2.0)
            y = (1 - gamma) * y + gamma * s_plus(y)
        return f_prime(y), y, self(y)


    def make_random_partitions(self, num_parts):
        groups = utils.partition(list(range(self.dim)), num_parts)
        return self.partition(groups)

    def make_scored_partitions(self, num_parts, score_fxn):
        dim = self.generator.shape[0]
        size = math.ceil(dim / num_parts)
        dim_scores = torch.tensor([dim_score(self, i) for i in range(dim)])
        sorted_scores = list(torch.sort(dim_scores, descending=True)[1].numpy())
        groups = [sorted_scores[i: i + size] for i in range(0, dim, size)]
        return self.partition(groups)



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
        return list(zip(groups, out_zonos))


    def reduce_simple(self, order, score='norm'):
        """ Does the simplest order reduction possible
        Keeps the (order-1) * dim largest 2-norm generatorumns
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

        return Zonotope(self.center, torch.cat([keep_gens, box_gens], dim=1),
                        max_order=self.max_order)



    # =============================================
    # =           Pushforward Operators           =
    # =============================================
    def map_linear(self, linear):
        new_center = linear(self.center)
        new_generator = linear.weight @ self.generator
        return Zonotope(new_center, new_generator, max_order=self.max_order)

    def map_conv(self, conv):
        new_center = conv(self.center.view(-1, *conv.input_shape))

        copy_layer = copy.copy(conv)
        copy_layer.weight.data = copy_layer.weight.abs().data
        copy_layer.bias = None

        new_generator = conv._conv_forward(self.generator.view(-1, *conv.input_shape),
                                           conv.weight.abs(), None)

        num_gens = self.generator.shape[1]
        return Zonotope(torch.flatten(new_center),
                        new_generator.view(-1, num_gens), max_order=self.max_order)


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


        new_zono = Zonotope(new_center, torch.cat([new_generator, new_cols], dim=1),
                            max_order=self.max_order)
        if self.max_order is not None and new_zono.order > self.max_order:
            new_zono = new_zono.reduce_simple(self.max_order, score='axalign')
        return new_zono


    def map_avgpool(self, layer):
        new_center = layer(self.center.view(-1, *layer.input_shape))
        new_gens = layer(self.generator.view(-1, *layer.input_shape))

        num_gens = self.generator.shape[1]
        return Zonotope(torch.flatten(new_center),
                        new_gens.view(-1, num_gens),
                        max_order=self.max_order)


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


    def _setup_relu_mip(self):
        """ Sets up the optimization program:
            min c1*z + c2*Relu(z) over this zonotope

        if apx_params is None, we return only a LOWER BOUND
        on the objective value
        Returns
            (opt_val, argmin x, argmin y)
        """
        mip_dict = self._encode_mip()
        model = mip_dict['model']


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


        model.update()

        return model, xs, all_relu_vars


    def solve_relu_mip(self, c1, c2, apx_params=None, verbose=False):
        # Setup the model (or load it if saved)
        if self.keep_mip:
            if self.relu_prog_model is None:
                self.relu_prog_model = self._setup_relu_mip()
            model, xs, relu_vars = self.relu_prog_model
        else:
            model, xs, relu_vars = self._setup_relu_mip()

        # Encode the parameters
        for k,v in (apx_params or {}).items():
            model.setParam(k, v)

        if verbose is False:
            model.setParam('OutputFlag', False)

        # Set the objective and optimize
        model.setObjective(gb.LinExpr(c1, xs) + gb.LinExpr(c2, relu_vars),
                           gb.GRB.MINIMIZE)
        model.update()
        model.optimize()

        if apx_params is not None:
            return model.ObjBound, model
        obj = model.objVal
        xvals = [_.x for _ in xs]

        return obj, xvals, model


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
        for z, g in zonos:
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






