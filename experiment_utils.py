""" Experiment helper files. Useful in bound comparisons """


###### UGLIEST IMPORT BLOCK EVER (gonna copy/paste this everywhere =P) ##########################

# Hacky relative import stuff
import os, sys
real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)
stcb_path = os.path.join(dir_path, 'scaling_the_convex_barrier')
sys.path.append(stcb_path)


# Packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import time
import pickle
import glob
import math
import copy

# My stuff
from abstract_domains import Hyperbox, Zonotope, Polytope
from dual_decompose import DecompDual
from lp_relaxation import LPRelax
from mip_verify import MIPVerify
from neural_nets import FFNet, PreactBounds, KWBounds, BoxInformedZonos
import train
import utilities as utils
from partitions import PartitionGroup



from scaling_the_convex_barrier.plnn.proxlp_solver.solver import SaddleLP
from scaling_the_convex_barrier.plnn.explp_solver.solver import ExpLP
from scaling_the_convex_barrier.plnn.network_linear_approximation import LinearizedNetwork
from scaling_the_convex_barrier.plnn.anderson_linear_approximation import AndersonLinearizedNetwork
from scaling_the_convex_barrier.tools.bounding_tools.anderson_cifar_bound_comparison import make_elided_models, dump_bounds
from scaling_the_convex_barrier.tools.bab_tools.model_utils import mnist_model, mnist_model_deep
from scaling_the_convex_barrier.plnn.modules import View, Flatten


####################################################

# ===============================================================================
# =           GLOBAL EXPERIMENT HELPERS                                         =
# ===============================================================================

def try_cache_clear():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def hbox_to_domain(hbox, network=None):
    # General method to convert Hyperbox inputs to the format used by stcb
    if network is not None:
        input_shape = network[0].input_shape
        shaper = lambda x: x.view(input_shape)
    else:
        shaper = lambda x: x.flatten()

    lbs = shaper(hbox.lbs)
    ubs = shaper(hbox.ubs)
    return torch.stack([lbs, ubs], dim=-1).unsqueeze(0)


def ovalnet_to_zonoinfo(ovalnet, bin_net, test_input):
    all_boxes = [Hyperbox(lb.flatten().data, ub.flatten().data) for (lb,ub)
                 in zip(ovalnet.lower_bounds, ovalnet.upper_bounds)]
    return BoxInformedZonos(bin_net, test_input, box_range=all_boxes).compute()


def get_device():
    if torch.cuda.is_available():
        return torch.device(type='cuda')
    else:
        return torch.device(type='cpu')


def get_best_naive_kw(bin_net, test_input):
    device = get_device()
    domain = hbox_to_domain(test_input, bin_net).to(device)

    # First get intermediate bounds
    elided_model = copy.deepcopy(bin_net)
    intermediate_net = SaddleLP([lay for lay in elided_model.to(device)])
    with torch.no_grad():
        intermediate_net.set_solution_optimizer('best_naive_kw', None)
        intermediate_net.define_linear_approximation(domain, no_conv=False,
                                                     override_numerical_errors=True)
    intermediate_ubs = intermediate_net.upper_bounds
    intermediate_lbs = intermediate_net.lower_bounds
    return intermediate_lbs, intermediate_ubs


def run_optprox(bin_net, test_input, use_intermed=True, return_model=False):
    device = get_device()
    # THEN RUN OPTPROX
    optprox_params = {
        'nb_total_steps': 400,
        'max_nb_inner_steps': 2,  # this is 2/5 as simpleprox
        'initial_eta': 1e0,
        'final_eta': 5e1,
        'log_values': False,
        'inner_cutoff': 0,
        'maintain_primal': True,
        'acceleration_dict': {
            'momentum': 0.3,  # decent momentum: 0.9 w/ increasing eta
        }
    }
    domain = hbox_to_domain(test_input, bin_net).to(device)
    optprox_net = SaddleLP([lay for lay in copy.deepcopy(bin_net).to(device)])

    optprox_start = time.time()
    with torch.no_grad():
        optprox_net.set_decomposition('pairs', 'KW')
        optprox_net.set_solution_optimizer('optimized_prox', optprox_params)
        if not use_intermed:
            optprox_net.define_linear_approximation(domain, no_conv=False)
            ub = optprox_net.upper_bounds[-1]
            lb = optprox_net.lower_bounds[-1]
        else:
            intermediate_lbs, intermediate_ubs = get_best_naive_kw(bin_net, test_input)
            optprox_net.build_model_using_bounds(domain, (intermediate_lbs, intermediate_ubs))
            lb, ub = optprox_net.compute_lower_bound()

    optprox_time = time.time() - optprox_start
    if return_model:
        return optprox_net, lb.cpu(), optprox_time
    return lb.cpu(), optprox_time


def run_explp(bin_net, test_input, use_intermed=True, return_model=False, num_iters=1000):
    device = get_device()

    # EXPLP
    explp_params = {
        "nb_iter": num_iters, #Using 1k iters as a benchmark
        'bigm': "init",
        'cut': "only",
        "bigm_algorithm": "adam",
        'cut_frequency': 450,
        'max_cuts': 12,
        'cut_add': 2,
        'betas': (0.9, 0.999),
        'initial_step_size': 1e-3,
        'final_step_size': 1e-6,
        "init_params": {
            "nb_outer_iter": 500,
            'initial_step_size': 1e-1,
            'final_step_size': 1e-3,
            'betas': (0.9, 0.999)
        },
    }
    domain = hbox_to_domain(test_input, bin_net)
    exp_net = ExpLP(copy.deepcopy(bin_net).to(device),
                     params=explp_params, use_preactivation=True, fixed_M=True)

    exp_start = time.time()
    with torch.no_grad():
        if not use_intermed:
            exp_net.define_linear_approximation(domain)
            ub = exp_net.upper_bounds[-1]
            lb = exp_net.lower_bounds[-1]
        else:
            intermediate_lbs, intermediate_ubs = get_best_naive_kw(bin_net, test_input)
            exp_net.build_model_using_bounds(domain, (intermediate_lbs, intermediate_ubs))
            lb, ub = exp_net.compute_lower_bound()
    exp_end = time.time()
    exp_time = exp_end - exp_start

    if return_model:
        return exp_net, lb.cpu().item(), exp_time
    return lb.item(), exp_time


def run_anderson_1cut(bin_net, test_input, use_intermed=True):
    device = get_device()
    domain = hbox_to_domain(test_input, bin_net).to(device)
    intermediate_lbs, intermediate_ubs = get_best_naive_kw(bin_net, test_input)

    domain.cpu()

    elided_model = copy.deepcopy(bin_net)
    elided_model_layers = [lay.cpu() for lay in elided_model]
    elided_model_layers[-1] = utils.negate_layer(elided_model_layers[-1])
    lp_and_grb_net = AndersonLinearizedNetwork(elided_model_layers, mode="lp-cut",
                                               n_cuts=1, cuts_per_neuron=True)
    lp_and_grb_start = time.time()
    if not use_intermed:
        lp_and_grb_net.define_linear_approximation(domain[0], n_threads=4)
        ub = lp_and_grb_net.upper_bounds[-1]
        lb = lp_and_grb_net.lower_bounds[-1]
        print(ub, lb)
    else:
        lp_and_grb_net.build_model_using_bounds(domain[0].cpu(), ([lbs[0].cpu() for lbs in intermediate_lbs],
                                                         [ubs[0].cpu() for ubs in intermediate_ubs]), n_threads=4)
        lb, ub = lp_and_grb_net.compute_lower_bound(ub_only=True)
    lp_and_grb_end = time.time()
    lp_and_grb_time = lp_and_grb_end - lp_and_grb_start
    lp_and_grb_lbs = -ub.item()
    return lp_and_grb_lbs, lp_and_grb_time


def run_lp(bin_net, test_input, use_intermed=True):
    device = get_device()
    domain = hbox_to_domain(test_input, bin_net).to(device)
    intermediate_lbs, intermediate_ubs = get_best_naive_kw(bin_net, test_input)

    elided_model = copy.deepcopy(bin_net).to(device)


    elided_model = elided_model.cpu()
    domain = domain.cpu()

    elided_model_layers = [lay for lay in elided_model]
    elided_model_layers[-1] = utils.negate_layer(elided_model_layers[-1])
    lp_and_grb_net = LinearizedNetwork(elided_model_layers)
    lp_and_grb_start = time.time()
    if not use_intermed:
        lp_and_grb_net.define_linear_approximation(domain[0], n_threads=4)
        ub = lp_and_grb_net.upper_bounds[-1]
        lb = lp_and_grb_net.lower_bounds[-1]

        print(ub, lb)
    else:

        lp_and_grb_net.build_model_using_bounds(domain[0].cpu(), ([lbs[0].cpu() for lbs in intermediate_lbs],
                                                         [ubs[0].cpu() for ubs in intermediate_ubs]), n_threads=4)
        lb, ub = lp_and_grb_net.compute_lower_bound(ub_only=True)
    lp_and_grb_end = time.time()
    lp_and_grb_time = lp_and_grb_end - lp_and_grb_start
    lp_and_grb_lbs = -ub
    return lp_and_grb_lbs.item(), lp_and_grb_time



# ==============================================================================
# =           MNIST EXPERIMENT HELPERS                                         =
# ==============================================================================

def setup_mnist_example(network, ex_id, eps, show=False, elide=False, normalize=None):
    # Sets up an example on the mnist dataset. Returns None if net is wrong on this example
    # Otherwise, returns (bin_net, Hyperbox) that we evaluate
    # By default clips into range [0,1]
    # normalize : if not None, is a


    valset = datasets.MNIST(root=train.DEFAULT_DATASET_DIR, train=False, download=True,
                            transform=transforms.ToTensor())
    device = get_device()
    x, y = valset[ex_id]
    x = x.to(device)

    if show:
        utils.show_grayscale(x)

    network.to(device)
    pred_label = network(x[None]).squeeze().max(dim=0)[1]

    if y != pred_label.item():
        return None, None
    if elide:
        bin_net = network.elide(y).to(device)
    else:
        bin_net = network.binarize(y, (y+1) % 10).to(device)


    test_input = Hyperbox.linf_box(x.flatten(), eps).clamp(0.0, 1.0).normalize(normalize)

    bin_net(x[None])
    return bin_net, test_input


def load_mnist_ffnet(pth=None):
    sequential = nn.Sequential(nn.Linear(784, 512), nn.ReLU(),
                               nn.Linear(512, 256), nn.ReLU(),
                               nn.Linear(256, 128), nn.ReLU(),
                               nn.Linear(128, 64), nn.ReLU(),
                               nn.Linear(64, 10))

    ffnet = FFNet(sequential)

    if pth is None:
        pth = os.path.join(dir_path, 'networks/mnist_ffnet.pth')

    ffnet.load_state_dict(torch.load(pth, map_location='cpu'))
    ffnet(torch.rand(1,1,28,28))# sets shapes
    return ffnet

def load_mnist_wide_net(pth=None):
    sequential = nn.Sequential(nn.Conv2d(1, 16, 4, stride=2, padding=1, bias=True), nn.ReLU(),
                               nn.Conv2d(16, 32, 4, stride=2, padding=1, bias=True), nn.ReLU(),
                               nn.Flatten(),
                               nn.Linear(32 * 7 * 7, 100, bias=True), nn.ReLU(),
                               nn.Linear(100, 10, bias=True))

    ffnet = FFNet(sequential)
    if pth is None:
        pth = os.path.join(dir_path, 'networks/mnist_wide.pth')
    ffnet.load_state_dict(torch.load(pth, map_location='cpu'))
    ffnet(torch.rand(1,1,28,28))# sets shapes
    return ffnet


def load_mnist_deep_net(pth=None):
    sequential = nn.Sequential(nn.Conv2d(1, 8, 4, stride=2, padding=1, bias=True), nn.ReLU(),
                               nn.Conv2d(8, 8, 3, stride=1, padding=1, bias=True), nn.ReLU(),
                               nn.Conv2d(8, 8, 3, stride=1, padding=1, bias=True), nn.ReLU(),
                               nn.Conv2d(8, 8, 4, stride=2, padding=1, bias=True), nn.ReLU(),
                               nn.Flatten(),
                               nn.Linear(8 * 7 * 7, 100, bias=True), nn.ReLU(),
                               nn.Linear(100, 10))
    ffnet = FFNet(sequential)
    if pth is None:
        pth = os.path.join(dir_path, 'networks/mnist_deep.pth')
    ffnet.load_state_dict(torch.load(pth, map_location='cpu'))
    ffnet(torch.rand(1,1,28,28))# sets shapes
    return ffnet

#####--------------------------- Decomp parameters here for various nets here -------------

def decomp_2d_MNIST_PARAMS(bin_net, test_input, return_obj=False, preact_bounds=None,
                           time_offset=0.0):
    device = get_device()
    bin_net.to(device)
    test_input.to(device)

    start_time = time.time()
    # Use best of KW/Naive to inform zonos

    if preact_bounds is None:
        lbs, ubs = get_best_naive_kw(bin_net, test_input)
        hboxes = [Hyperbox(lb.flatten(), ub.flatten()) for (lb, ub) in zip(lbs, ubs)]
        preact_bounds = BoxInformedZonos(bin_net, test_input, box_range=hboxes).compute()

    # Define partition object
    partition_obj = PartitionGroup(None, style='fixed_dim', partition_rule='similarity',
                                   partition_dim=2)

    # Main dual object and ascent procedure
    decomp = DecompDual(bin_net, test_input, Zonotope, 'partition', zero_dual=False,
                         partition=partition_obj, preact_bounds=preact_bounds)

    optim_obj = optim.Adam(decomp.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.MultiplicativeLR(optim_obj, lambda epoch: 0.75, last_epoch=- 1, verbose=False)
    for i in range(10):
        last_val = decomp.dual_ascent(100, verbose=200, optim_obj=optim_obj, iter_start=i * 100)
        scheduler.step()

    if return_obj:
        return decomp
    return last_val.item(), time_offset + time.time() - start_time




# ===============================================================================
# =           CIFAR EXPERIMENT HELPERS                                          =
# ===============================================================================


def setup_cifar_example(network, ex_id, eps, elide=False, normalize=None):
    # Sets up an example on the cifar dataset. Returns None if net is wrong on this example
    # Otherwise, returns (bin_net, Hyperbox) that we evaluate
    # By default clips into range [0,1]
    if normalize is None:
        standard_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.225, 0.225, 0.225])
        valset = datasets.CIFAR10(root=train.DEFAULT_DATASET_DIR, train=False, download=True,
                                transform=transforms.Compose([transforms.ToTensor(),
                                                              standard_normalize]))
    else:
        valset = datasets.CIFAR10(root=train.DEFAULT_DATASET_DIR, train=False, download=True,
                                transform=transforms.Compose([transforms.ToTensor()]))

    device = get_device()
    x, y = valset[ex_id]
    x = x.to(device)

    network.to(device)
    pred_label = network(x[None]).squeeze().max(dim=0)[1]

    if y != pred_label.item():
        return None, None

    if elide:
        bin_net = network.elide(y).to(device)
    else:
        bin_net = network.binarize(y, (y+1) % 10).to(device)

    test_input = Hyperbox.linf_box(x.flatten(), eps).normalize(normalize)
    bin_net(x[None])
    return bin_net, test_input


def load_cifar_sgd(pth=None):
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )

    ffnet = FFNet(model)

    if pth is None:
        pth = os.path.join(dir_path, 'networks/cifar_sgd.pth')

    ffnet.load_state_dict(torch.load(pth, map_location='cpu'))
    ffnet(torch.rand(1,3,32,32))# sets shapes
    return ffnet


def load_cifar_madry(pth=None):
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )

    ffnet = FFNet(model)

    if pth is None:
        pth = os.path.join(dir_path, 'networks/cifar_madry.pth')

    ffnet.load_state_dict(torch.load(pth, map_location='cpu'))
    ffnet(torch.rand(1,3,32,32))# sets shapes
    return ffnet

def load_mnist_wide_net(pth=None):
    sequential = nn.Sequential(nn.Conv2d(1, 16, 4, stride=2, padding=1, bias=True), nn.ReLU(),
                               nn.Conv2d(16, 32, 4, stride=2, padding=1, bias=True), nn.ReLU(),
                               nn.Flatten(),
                               nn.Linear(32 * 7 * 7, 100, bias=True), nn.ReLU(),
                               nn.Linear(100, 10, bias=True))

    ffnet = FFNet(sequential)
    if pth is None:
        pth = os.path.join(dir_path, 'networks/mnist_wide.pth')
    ffnet.load_state_dict(torch.load(pth, map_location='cpu'))
    ffnet(torch.rand(1,1,28,28))# sets shapes
    return ffnet




def decomp_2d_CIFAR_PARAMS(bin_net, test_input, return_obj=False, preact_bounds=None,
                           time_offset=0.0):
    device = get_device()
    bin_net.to(device)
    test_input.to(device)

    start_time = time.time()
    # Use best of KW/Naive to inform zonos

    if preact_bounds is None:
        lbs, ubs = get_best_naive_kw(bin_net, test_input)
        hboxes = [Hyperbox(lb.flatten(), ub.flatten()) for (lb, ub) in zip(lbs, ubs)]
        preact_bounds = BoxInformedZonos(bin_net, test_input, box_range=hboxes).compute()

    # Define partition object
    input_shapes = [l.input_shape for l in bin_net.net]

    partition_obj = PartitionGroup(None, style='fixed_dim', partition_rule='spatial',
                                   partition_dim=2, input_shapes=input_shapes)

    # Main dual object and ascent procedure
    decomp = DecompDual(bin_net, test_input, Zonotope, 'partition', zero_dual=False,
                         partition=partition_obj, preact_bounds=preact_bounds)

    optim_obj = optim.Adam(decomp.parameters(), lr=2.5e-3)
    scheduler = optim.lr_scheduler.MultiplicativeLR(optim_obj, lambda epoch: 0.5, last_epoch=- 1, verbose=False)
    for i in range(5):
        last_val = decomp.dual_ascent(200, verbose=50, optim_obj=optim_obj, iter_start=i * 200)
        scheduler.step()

    if return_obj:
        return decomp

    return last_val.item(), time_offset + time.time() - start_time
