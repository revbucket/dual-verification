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
import scipy.optimize
import sys

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
        self.dim = self.lbs.numel()

    def cuda(self):
        self.lbs = self.lbs.cuda()
        self.ubs = self.ubs.cuda()
        self.center = self.center.cuda()
        self.rad = self.rad.cuda()
        return self

    def cpu(self):
        self.lbs = self.lbs.cpu()
        self.ubs = self.ubs.cpu()
        self.center = self.center.cpu()
        self.rad = self.rad.cpu()
        return self

    def twocol(self):
        return torch.stack([self.lbs, self.ubs]).T

    def __getitem__(self, items):
        new_lbs = self.lbs.__getitem__(items)
        new_ubs = self.ubs.__getitem__(items)
        return Hyperbox(new_lbs, new_ubs)

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


    def solve_lp(self, obj: torch.Tensor, bias=0.0, get_argmin: bool = False):
        """ Solves linear programs like min <obj, x> over hyperbox
        ARGS: obj : objective vector
              get_opt_point : if True, returns optimal point too
        RETURNS: either optimal_value or (optimal_value, optimal point)
        """
        # Do this via centers and rads
        signs = torch.sign(obj)
        opt_point = F.relu(signs) * self.lbs + F.relu(-signs) * self.ubs
        opt_val = obj @ opt_point + bias
        if get_argmin:
            return opt_val, opt_point
        return opt_val


    def solve_relu_program(self, lin_obj, relu_obj, get_argmin: bool= False):
        """ Solves ReLU program like min lin_obj @x + relu_obj @Relu(x) over hyperbox
        ARGS:
            lin_obj : objective vector for linear term
            relu_obj : objective vector for relu term
        RETURNS: either optimal_value or (optimal_value, optimal_point)
        """

        argmin = torch.clone(self.center)

        # Handle lb >0 case
        pos_idxs = self.lbs > 0
        sum_obj = lin_obj + relu_obj
        argmin[pos_idxs] = torch.where(sum_obj[pos_idxs] >= 0, self.lbs[pos_idxs], self.ubs[pos_idxs])

        # Handle ub < 0 case
        neg_idxs = self.ubs < 0
        argmin[neg_idxs] = torch.where(lin_obj[neg_idxs] >= 0, self.lbs[neg_idxs], self.ubs[neg_idxs])

        # Handle ambiguous case
        ambig_idxs = (~(pos_idxs + neg_idxs)).nonzero()

        argmin_ambig = argmin[ambig_idxs]
        lbs_ambig = self.lbs[ambig_idxs]
        ubs_ambig = self.ubs[ambig_idxs]
        lin_ambig = lin_obj[ambig_idxs]
        relu_ambig = relu_obj[ambig_idxs]

        eval_at_l = lin_ambig * lbs_ambig
        eval_at_u = (lin_ambig + relu_ambig) * ubs_ambig
        eval_at_0 = torch.zeros_like(eval_at_l)
        min_idxs = torch.stack([eval_at_l, eval_at_u, eval_at_0]).min(dim=0)[1]
        argmin[ambig_idxs[min_idxs == 0]] = lbs_ambig[min_idxs == 0]
        argmin[ambig_idxs[min_idxs == 1]] = ubs_ambig[min_idxs == 1]
        argmin[ambig_idxs[min_idxs == 2]] = 0.0

        # Compute optimal value and return
        opt_val = lin_obj @ argmin + relu_obj @ torch.relu(argmin)
        if get_argmin:
            return opt_val, argmin
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
        self.relu_prog_model = None
        self.past_rp_solution = None

        self._compute_state()


    def _compute_state(self):
        hbox = self.as_hyperbox()
        self.lbs = hbox.lbs
        self.ubs = hbox.ubs
        self.rad = (self.ubs - self.lbs) / 2.0
        self.dim = self.lbs.numel()
        self.gensize = self.generator.shape[1]
        self.order = self.gensize / self.dim


    def __call__(self, y):
        return self.center + self.generator @ y

    def __getitem__(self, items):
        new_center = self.center.__getitem__(items)
        new_generator = self.generator.__getitem__(items)
        return Zonotope(new_center, new_generator, max_order=self.max_order)

    def __repr__(self):
        return "Zonotope(%s,%s)" % (self.dim, self.gensize)


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
        self.center = self.center.cuda()
        self.generator = self.generator.cuda()
        self.lbs = self.lbs.cuda()
        self.ubs = self.ubs.cuda()
        self.rad = self.rad.cuda()
        return self

    def cpu(self):
        self.center = self.center.cpu()
        self.generator = self.generator.cpu()
        self.lbs = self.lbs.cpu()
        self.ubs = self.ubs.cpu()
        self.rad = self.rad.cpu()
        return self

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

    def solve_lp(self, obj: torch.Tensor, get_argmin: bool = False, bias=0.0):
        # RETURNS: either optimal_value or (optimal_value, optimal point)
        center_val = self.center @ obj
        opt_signs = -torch.sign(self.generator.T @ obj)
        opt_point = self.center + self.generator @ opt_signs
        opt_val = opt_point @ obj + bias
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
        yvars = model.addVars(range(self.gensize), lb=-1.0-eps, ub=1.0+eps)
        yvars = [yvars[_] for _ in range(self.gensize)]
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

    def fw_rp(self, c1, c2, num_iter=1000, rand_init=True, num_init=1):
        zeros = torch.zeros_like(c2)

        def sub_argmin(x):
            # Finds the s in Z that minimizes s@grad_f(x)
            grad_f = c1 + torch.where(x > 0, c2, zeros)
            return self.center + self.generator @ torch.sign(-grad_f @ self.generator)

        if rand_init:
            x = self(torch.rand_like(self.generator[0]) *2 - 1.0)
        else:
            x = self.center

        for step in range(num_iter):
            gamma = 2 / float(step + 2.0)
            x = (1 -gamma) * x + gamma * sub_argmin(x)
        return c1@x + c2 @ x.relu(), x




    def solve_relu_simplex(self, c1, c2, iters=3):
        best_o, best_y = float('inf'), None

        for _ in range(iters):
            y = 2 * (torch.rand(self.generator.shape[1]) > 0.5).float() - 1
            z = self(y)
            o = c1 @ z + c2 @ F.relu(z)
            G, b = self.generator, self.center

            while True:
                z = -2*G*y + (G@y + b)[None].T
                n = c1 @ z + c2 @ F.relu(z)
                m, i = n.min(0)
                if o <= m: break
                ya = y * (-2*(o - n > 0) + 1)
                za = self(ya)
                oa = c1 @ za + c2 @ F.relu(za)
                if oa <= m:
                    o, y = oa, ya
                else:
                    o = m
                    y[i] *= -1

            if o < best_o:
                best_o, best_y = o, y

        return best_o, best_y


    def solve_relu_fw(self, c1, c2, num_iter=10000):
        return self.fw(lambda x: c1 @ x + c2 @ F.relu(x), num_iter)


    def solve_relu_lbfgsb(self, c1, c2, iters=1, start="simplex", simplex_iters=None):
        g = self.generator.shape[1]
        # NOTE: this is important, otherwise we'll end up backpropping
        # through the outer optimization graph
        c1, c2 = c1.detach(), c2.detach()

        def f(y):
            x = self(y)
            return c1 @ x + c2 @ F.relu(x)

        def value_and_grad(y):
            y = torch.from_numpy(y).float().detach().requires_grad_()
            out = f(y)
            out.backward()
            return out.item(), y.grad.numpy().astype(np.double)

        best_o, best_y = float('inf'), None

        for _ in range(iters):
            if start == "zero":
                y0 = np.zeros(g)
                assert iters == 1
            elif start == "simplex":
                kwargs = {"iters": simplex_iters} if simplex_iters is not None else {}
                sopt, sy = self.solve_relu_simplex(c1, c2, **kwargs)
                y0 = sy.double().numpy()
            elif start == "random":
                y0 = 2 * (np.random.rand(g) > 0.5) - 1,
            else:
                raise NotImplementedError()

            opt = scipy.optimize.minimize(
                value_and_grad,
                y0,
                method="L-BFGS-B",
                jac=True,
                bounds=scipy.optimize.Bounds(np.full(g, -1), np.full(g, 1)),
            )

            if opt.fun < best_o:
                best_o, best_y = opt.fun, opt.x

        return torch.tensor(best_o), torch.from_numpy(best_y).float().detach()

    def make_random_partitions(self, num_parts):
        groups = utils.partition(list(range(self.dim)), num_parts)
        return self.partition(groups)

    def make_random_partitions_dim(self, partition_dim):
        num_parts = math.ceil(self.dim / partition_dim)
        return self.make_random_partitions(num_parts)



    def make_scored_partitions(self, num_parts, score_fxn):
        dim = self.generator.shape[0]
        size = math.ceil(dim / num_parts)
        dim_scores = torch.tensor([dim_score(self, i) for i in range(dim)]).to(self.center.device)
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

        #groups = utils.complete_partition(groups, self.lbs.numel())

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

        input_shape = conv.input_shape
        new_center = conv(self.center.view(-1, *input_shape)).view(-1)
        generator = self.generator.T.view((self.gensize,) + input_shape)
        new_generator = F.conv2d(generator, weight=conv.weight, bias=None, stride=conv.stride,
                                 padding=conv.padding, dilation=conv.dilation, groups=conv.groups)

        new_generator = new_generator.view(self.gensize, -1).T
        return Zonotope(new_center, new_generator, max_order=self.max_order)


    def map_relu(self, extra_box=None):
        """ Remember how to do this...
            Want to minimize the vertical deviation on
            |lambda * (c + Ey) - ReLU(c+Ey)| across c+Ey in [l, u]

            if extra_box is not None, is a hyperbox with extra bounds we know to be valid
        """
        #### SOME SORT OF BUG HERE --- NOT PASSING SANITY CHECKS
        new_center = torch.clone(self.center)
        new_generator = torch.clone(self.generator)

        if extra_box is None:
            lbs = self.lbs
            ubs = self.ubs
        else:
            lbs = torch.max(torch.stack([self.lbs, extra_box.lbs]), dim=0)[0]
            ubs = torch.min(torch.stack([self.ubs, extra_box.ubs]), dim=0)[0]



        on_neurons = lbs > 0
        off_neurons = ubs < 0
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
        scale = ubs[unstable] / (ubs[unstable] - lbs[unstable])
        offset = -lbs[unstable] * scale / 2.0

        new_generator[unstable] *= scale.view(-1, 1) # 1
        new_center[unstable] *= scale                # 2
        new_center[unstable] += offset               # 3

        new_cols = torch.zeros_like(lbs.view(-1, 1).expand(-1, unstable.sum())) # 4
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

    def _encode_mip2(self, box_bounds=None):
        eps =1e-6
        model = gb.Model()
        ys = model.addMVar(self.gensize, lb=-1, ub=1, name='y')
        model.update()

        if box_bounds is None:
            lbs = self.lbs
            ubs = self.ubs
        else:
            lbs = torch.max(self.lbs, box_bounds.lbs)
            ubs = torch.min(self.ubs, box_bounds.ubs)

        xs = model.addMVar(self.dim, lb=lbs - eps, ub=ubs + eps, name='x')


        model.addConstr(self.generator.cpu().numpy() @ ys + self.center.cpu().numpy() == xs)
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
        unc_idxs = ((self.lbs * self.ubs) < 0).nonzero().squeeze(0)
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

        return model, xs, ys, all_relu_vars

    def _setup_relu_mip2(self, box_bounds=None):
        """ Sets up the optimization program:
            min c1*z + c2*Relu(z) over this zonotope

        if apx_params is None, we return only a LOWER BOUND
        on the objective value
        Returns
            (opt_val, argmin x, argmin y)
        """
        mip_dict = self._encode_mip2(box_bounds=box_bounds)
        model = mip_dict['model']


        xs, ys = mip_dict['xs'], mip_dict['ys']

        # Now add ReLU constraints, using the big-M encoding
        unc_idxs = ((self.lbs * self.ubs) < 0).nonzero().flatten().cpu()

        relu_vars = model.addMVar(len(unc_idxs), lb=0, ub=self.ubs[unc_idxs], name='relu')
        zs = model.addMVar(len(unc_idxs), vtype=gb.GRB.BINARY, name='z')

        if len(unc_idxs) > 1:
            model.addConstr(relu_vars >= xs[unc_idxs])

            # Maybe diagflats aren't best here...
            model.addConstr(relu_vars <= np.diagflat(self.ubs[unc_idxs].detach().cpu().numpy()) @ zs)
            model.addConstr(relu_vars + self.lbs[unc_idxs].squeeze().detach().cpu().numpy() <=
                        xs[unc_idxs] + np.diagflat(self.lbs[unc_idxs].detach().cpu().numpy()) @ zs)
        else:
            for i, idx in enumerate(unc_idxs):
                model.addConstr(relu_vars[i] >= xs[idx])
                model.addConstr(relu_vars[i] <= self.ubs[idx].item() * zs[i])
                model.addConstr(relu_vars[i] <= xs[idx] - (1- zs[i]) * self.lbs[idx].item())


        model.update()

        return model, xs, ys, relu_vars, unc_idxs


    def solve_relu_mip(self, c1, c2, apx_params=None, verbose=False, start=None, start_kwargs={},
                       box_bounds=None):
        # Setup the model (or load it if saved)
        if self.keep_mip:
            if self.relu_prog_model is None:
                self.relu_prog_model = self._setup_relu_mip2(box_bounds)
            model, xs, ys, relu_vars, unc_idxs = self.relu_prog_model
        else:
            model, xs, ys, relu_vars, unc_idxs = self._setup_relu_mip2(box_bounds)



        # Encode the parameters
        apx_params = apx_params if (apx_params is not None) else {}
        apx_params['OutputFlag'] = verbose
        for k,v in apx_params.items():
            model.setParam(k, v)

        # Set the objective and optimize
        c1 = c1.detach().cpu().numpy()
        c2 = c2[unc_idxs].squeeze().detach().cpu().numpy()
        try:
            model.setObjective(c1 @ xs + c2 @ relu_vars, gb.GRB.MINIMIZE)
        except Exception as err:
            # In the case that |unc_idxs| == 1
            model.setObjective(c1 @ xs + c2 * relu_vars, gb.GRB.MINIMIZE)

        if start == 'prev' and self.past_rp_solution is not None:
            for var, past_val in zip(model.getVars(), self.past_rp_solution):
                var.Start = past_val

        model.update()
        model.optimize()

        self.past_rp_solution = [_.X for _ in model.getVars()]


        xvals = torch.tensor([_.x for _ in xs], device=self.center.device)
        yvals = torch.tensor([_.x for _ in ys], device=self.center.device)

        return model.ObjBound, xvals, yvals, model


    def bound_relu_sdp(self, c1, c2):
        import mosek.fusion as mf
        c1, c2 = c1.detach().numpy().tolist(), c2.detach().numpy().tolist()
        b, G = self.center.numpy().tolist(), self.generator.numpy().tolist()
        n, m = self.dim, self.generator.shape[1]
        E = mf.Expr

        M = mf.Model("reluprog")

        X = M.variable(mf.Domain.inPSDCone(1+m+n))

        Xx = X.slice([1, 0], [m+1, 1])
        Xz = X.slice([m+1, 0], [m+n+1, 1])
        Xxx = X.slice([1, 1], [m+1, m+1])
        Xxz = X.slice([1, m+1], [m+1, m+n+1])
        Xzz = X.slice([m+1, m+1], [m+n+1, m+n+1])

        ge0 = mf.Domain.greaterThan(0.0)
        eq0 = mf.Domain.equalsTo(0.0)

        y = E.add(E.mul(G, Xx), b)
        M.constraint(Xz, ge0)
        M.constraint(E.sub(Xz, y), ge0)
        M.constraint(E.sub(Xzz.diag(), E.add(E.mulDiag(G, Xxz), E.mulElm(Xz, [[_] for _ in b]))), eq0)
        M.constraint(E.sub(1, Xxx.diag()), ge0)
        M.constraint(E.sub(X.index(0, 0), 1), eq0)

        M.objective(
            mf.ObjectiveSense.Minimize,
            E.add(E.dot(c1, y), E.dot(c2, Xz))
        )

        M.setLogHandler(sys.stdout)
        M.solve()

        return M.primalObjValue()


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
        for g, z in zonos:
            this_out = z.solve_relu_mip(c1[g], c2[g])
            outputs.append(this_out[0])
            opt_point[g] = torch.tensor(this_out[1], device=self.center.device)

        return sum(outputs), opt_point


    # =======================================================
    # =           2 Dimensional Partition blocks            =
    # =======================================================

    def enumerate_vertices_2d(self, collect_crossings=True):
        # Returns (2*gensize, 2) tensor of vertices for this zono
        assert self.dim == 2
        signs = -torch.sign(self.generator[0])
        signs[signs == 0] = 1.0
        left_point = self(signs)
        # Sort and orient the slopes
        slopes = self.generator[1] / self.generator[0]
        sorted_idxs = torch.sort(slopes)[1]
        orient_gen = self.generator * -signs.view(1,-1)
        shuffle_gen = orient_gen[:, sorted_idxs]
        cumsum_gen = 2 * shuffle_gen.cumsum(dim=1)
        pos_vs = left_point.view(1, -1) + cumsum_gen.T
        neg_vs = 2 * self.center.view(1, -1) - pos_vs

        vs = torch.cat([pos_vs, neg_vs])
        # Collecting coordinate-axes crossing points
        if collect_crossings is True:
            crossings = self.all_cross_2d(vs)
            if crossings is not None:
                vs = torch.cat([vs, crossings])
        return vs

    def crossing_vs_2d(self, vs, dim=0):
        if self.lbs[dim] * self.ubs[dim] > 0:
            return None
        num_vs = vs.shape[0]
        start_idxs = ((vs[1:,dim] * vs[:-1,dim]) < 0).nonzero().flatten()
        if start_idxs.numel() == 1:
            start_idxs = torch.cat([start_idxs, -torch.ones_like(start_idxs).long()])

        point_idxs =torch.stack([start_idxs, ((start_idxs + 1) % vs.shape[0])], dim=1)
        points = vs[point_idxs]
        starts = points[:,0,:]
        xs = points[:,:,dim]
        lens = xs @ torch.tensor([-1.0,1.0], device=self.center.device)
        diffs = points[:,1,:] - points[:,0,:]
        interp_factor = -xs[:,0] / lens
        offsets = diffs * interp_factor.view(-1, 1)
        mids = starts + offsets
        return mids

    def all_cross_2d(self, vs):
        output = []
        output.append(self.crossing_vs_2d(vs, dim=0))
        output.append(self.crossing_vs_2d(vs, dim=1))
        # And maybe add origin in there...

        if output[0] is not None and torch.prod(output[0][:,1]) < 0:
            output.append(torch.zeros_like(self.center).view(1, -1))
        noneless_output = [_ for _ in output if _ is not None]
        if len(noneless_output) > 0:
            return torch.cat(noneless_output)
        return None


    def relu_program_2d(self, c1, c2, vs=None):
        """ Returns (min_val, argmin) for 2d relu program
        NOTE: does not consider axes vals!
        """
        vs = vs if (vs is not None) else self.enumerate_vertices_2d(collect_crossings=True)
        vals = vs @ c1 + torch.relu(vs) @ c2
        min_val, argmin_idx = torch.min(vals, dim=0)
        return min_val, vs[argmin_idx]


    def batch_2d_partition(self, groups=None):
        """ Enumerates vertices based on groups
        ARGS:
            groups: is a tensor of [[group1.x, group1.x], ...]
        RETURNS:
            tensor of shape (num_groups, 2, 2*gensize)
        """
        assert self.dim % 2 == 0
        if groups is None:
            groups = torch.rand(self.dim).sort()[1].view(-1, 2)
        xs = groups[:,0]
        ys = groups[:,1]
        center, generator = self.center, self.generator

        signs = torch.sign(generator[xs,:])
        signs[signs == 0] = 1 # arbitrary resolve zeros
        sign_gen = torch.ones_like(generator)
        sign_gen[xs,:] = signs
        sign_gen[ys,:] = signs

        lefts = center - (generator * sign_gen).sum(dim=1)

        # Slopes now...
        slopes = generator[ys, :] / generator[xs,:]
        sorted_idxs = torch.sort(slopes, dim=1)[1] # (groups, gensize)

        # And now split for xs, ys (easier to reason about gather and such)
        orient_gen = sign_gen * generator
        shuffle_xs = orient_gen[xs,:].gather(1, sorted_idxs)
        shuffle_ys = orient_gen[ys,:].gather(1, sorted_idxs)

        cumsum_xs = 2 * shuffle_xs.cumsum(dim=1)
        cumsum_ys = 2 * shuffle_ys.cumsum(dim=1)

        pos_vx = lefts[xs].view(-1, 1) + cumsum_xs
        pos_vy = lefts[ys].view(-1, 1) + cumsum_ys

        neg_vx = 2 * center[xs].view(-1, 1) - pos_vx
        neg_vy = 2 * center[ys].view(-1, 1) - pos_vy


        # Output is (#Groups, 2, #vertices)
        return torch.stack([torch.cat([pos_vx, neg_vx], dim=1),
                            torch.cat([pos_vy, neg_vy], dim=1)], dim=1)

    def batch_2d_relu_program(self, c1, c2, groups=None, group_vs=None):
        """ Solves relu program by partition into groups of 2d
            e.g. min c1*z + c2*relu(z)
        ARGS:
            c1,c2: tensor for objective vector
            groups: groups arg to be fed into batch_2d_partition
            group_vs: if not None is like the output of batch_2d_partition
        RETURNS:
            min_val, argmin
        """
        if group_vs is None:
            group_vs = self.batch_2d_partition(groups=groups)

        obj_vals = (c1[groups].unsqueeze(-1) * group_vs +
                    c2[groups].unsqueeze(-1) * torch.relu(group_vs))
        mins, min_idxs = obj_vals.sum(dim=1).min(dim=1)

        argmin = torch.zeros_like(self.center)
        argmin[groups[:,0]] = torch.gather(group_vs[:,0,:], 1, min_idxs.view(-1, 1)).squeeze()
        argmin[groups[:,1]] = torch.gather(group_vs[:,1,:], 1, min_idxs.view(-1, 1)).squeeze()
        return mins.sum(), argmin


    # LOOP methods (not as fast, but has crossings)
    def loop_batch_zono_vs(self, groups=None):
        # Set up groups
        outputs = []
        if groups is None:
            groups = torch.rand_like(self.dim).sort()[1].view(-1, 2)
        for group in groups:
            subzono = Zonotope(self.center[group], self.generator[group,:])
            outputs.append(subzono.enumerate_vertices_2d(collect_crossings=True))



        """ HACK HERE:
            Because of the crossings, the groups no longer have same
            number of vertices. Ideally I want to shape them like
            outputs.shape == (num_groups, 2, #vertices)
            But #vertices differs so I will just pad with the first
            few columns (rows here) of each
        """
        # padding block
        max_vs = max([_.shape[0] for _ in outputs])
        for i, el in enumerate(outputs):
            num_to_pad = max_vs - el.shape[0]
            if num_to_pad == 0:
                continue
            pad = el[:num_to_pad, :]
            outputs[i] = torch.cat([el, pad])




        return torch.stack([_.T for _ in outputs], dim=0)

        # return outputs

    def loop_zono_rp(self, c1, c2, groups=None, loop_outs=None):

        def rp_vertices_2d(vs, c1, c2):
            vals = vs @ c1 + torch.relu(vs) @ c2
            min_val, argmin_idx = torch.min(vals, dim=0)
            return min_val, vs[argmin_idx]

        if groups is None:
            groups = torch.rand(self.dim).sort()[1].view(-1, 2)

        if loop_outs is not None:
            loop_outs = self.loop_batch_zono_vs(groups=groups)

        argmin = torch.zeros_like(self.center)
        min_val = 0.0
        for output, group in zip(loop_outs, groups):
            sub_min, sub_argmin = rp_vertices_2d(output, c1[group], c2[group])
            min_val += sub_min
            argmin[group] = sub_argmin
        return min_val, argmin

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

        return Hyperbox(torch.Tensor(lbs, device=self.center.device),
                        torch.Tensor(ubs, device=self.center.device))






