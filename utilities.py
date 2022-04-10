""" Utilities that are useful in general """
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import math
import os
import numpy as np
import threading
# =========================================================
# =           Constructors and abstract classes           =
# =========================================================

class ParameterObject:
    """ Classes that inherit this just hold named arguments.
        Can be extended, but otherwise works like a namedDict
    """
    def __init__(self, **kwargs):
        self.attr_list = []
        assert 'attr_list' not in kwargs
        for k,v in kwargs.items():
            setattr(self, k, v)
            self.attr_list.append(k)


def no_grad(f):
    def dec_fxn(*args, **kwargs):
        with torch.no_grad():
            return f(*args, **kwargs)
    return dec_fxn


# ========================================================
# =           Other helpful methods                      =
# ========================================================

def safe_open(path, style='wb'):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, style)


def negate_layer(layer):
    """ Replaces the weights and biases of a layer with their negatives
        NOTE: This MODIFIES THE LAYER (so be careful to copy!)
    """
    assert isinstance(layer, (nn.Linear, nn.Conv2d))

    layer.weight.data = -1 * layer.weight.data
    if layer.bias != None:
        layer.bias.data = -1 * layer.bias.data
    return layer


def add_flattens(network):
    """ Given an iterable of layers, builds a nn.Sequential with
        flattens inserted between each conv and linear layer
    """
    new_network = [network[0]]
    netlist = [layer for layer in network]
    last_affine = network[0]
    for i in range(1, len(netlist)):
        layer = netlist[i]
        if isinstance(layer, nn.Linear) and isinstance(last_affine, nn.Conv2d):
            new_network.append(nn.Flatten())
        new_network.append(layer)

        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            last_affine = layer
    return nn.Sequential(*netlist)


def tensorfy(x):
    if isinstance(x, torch.Tensor):
        return x
    else:
        return torch.Tensor(x)

def conv_indexer(input_shape):
    """ Makes dicts that map from:
            (C,H,W) -> index list
            index_list -> (C,H,W)
    """
    if len(input_shape) == 4:
        input_shape = input_shape[1:]
    C, H, W = input_shape
    numel = C * H * W
    idx_to_tup = {}
    i = 0
    for c in range(C):
        for h in range(H):
            for w in range(W):
                idx_to_tup[i] = (c, h, w)
                i +=1
    tup_to_idx = {v: k for k, v in idx_to_tup.items()}
    return tup_to_idx, idx_to_tup

def conv_output_shape(conv, input_shape=None):
    if input_shape is None:
        input_shape = conv.input_shape
    c, h, w = input_shape
    out_h = math.floor((h + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) -1) / conv.stride[0] + 1)
    out_w = math.floor((w + 2 * conv.padding[1] - conv.dilation[1] * (conv.kernel_size[1] - 1) -1) / conv.stride[1] + 1)
    return (conv.out_channels, out_h, out_w)

def conv_transpose_shape(conv, input_shape=None):
    """ When trying to transpose a given conv operator, returns the shape given
        by blindly applying conv2d_transpose
    """
    c, h, w = conv_output_shape(conv, input_shape=input_shape)
    s0, s1 = conv.stride
    p0, p1 = conv.padding
    d0, d1 = conv.dilation
    k0, k1 = conv.kernel_size

    out_h = (h - 1) * s0 - 2 * p0 + d0 *( k0 - 1) + 1
    out_w = (w - 1) * s1 - 2 * p1 + d1 * (k1 - 1) + 1
    return (conv.in_channels, out_h, out_w)

    # Maps output -> input (useful for determining padding)

def conv_to_lin_weight(conv, input_shape=None):
    """ Converts a convolutional layer to matrix ... super slow, but w/e"""
    if input_shape is None:
        input_shape = conv.input_shape
    output_numel = np.prod(conv_output_shape(conv, input_shape=input_shape))

    class grad_enabled_thread(threading.Thread):
        """ Hacky threaded way to escape the 'no_grad' context """
        def __init__(self, conv, output_numel, input_shape, output):
            threading.Thread.__init__(self)
            self.conv = conv
            self.output_numel = output_numel
            self.input_shape = input_shape
            self.output = output
        def run(self):
            x = torch.zeros((self.output_numel,) + self.input_shape).requires_grad_(True)
            x = x.to(self.conv.weight.device)
            self.conv(x).view(self.output_numel, -1).diag().sum().backward()
            self.output.append(x.grad.view(self.output_numel, -1).data)

    output = []
    thread1 = grad_enabled_thread(conv, output_numel, input_shape, output)
    thread1.start()
    thread1.join()
    return output[0]

def get_weight(layer, stack=True):
    """ Better access method to get weight of a layer"""
    if isinstance(layer, nn.Linear):
        weight = layer.weight
    else:
        weight = conv_to_lin_weight(layer)
    if stack:
        return torch.stack([weight, -weight], dim=0)
    return weight

def get_bias(layer):
    """Accompanying method for biases that works for linear/conv layers"""
    if layer.bias is None:
        return 0.0
    elif isinstance(layer, nn.Linear):
        return layer.bias
    elif isinstance(layer, nn.Conv2d):
        output_shape = conv_output_shape(layer)
        return layer.bias.view(-1, 1, 1).expand(output_shape).flatten()
    else:
        raise NotImplementedError()

def flatten(lol):
    output = []
    def subflatten(sublist, output=output):
        for el in sublist:
            if hasattr(el, '__iter__'):
                subflatten(el)
            else:
                output.append(el)
    subflatten(lol)
    return output


def conv_view(els, shape):
    """ Maps 1d list into shape """
    if len(shape) == 4:
        shape = shape[1:]

    C, H, W = shape
    rows = [els[i:i+W] for i in range(0, len(els), W)]
    channels = [rows[i: i+H] for i in range(0, len(rows), H)]
    return channels




def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

def complete_partition(groups, total):
    """ If given an incomplete partition, creates the indices needed
        to make it a full partition (and sorts each subgroup)
    """
    solo_group = isinstance(groups[0], int)
    total_groupsize = len(groups) if solo_group else sum(len(_) for _ in groups)
    if total_groupsize < total:
        if solo_group:
            all_idxs = set(groups)
            groups = [groups]
        else:
            all_idxs = set(groups[0]).union(groups[1:])
        outgroup = []
        for i in range(total):
            if i not in all_idxs:
                outgroup.append(i)

        groups = [sorted(_) for _ in groups] + [outgroup]
    else:
        groups = [sorted(_) for _ in groups]

    return groups


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

def get_runner_up_idx(model, x, y):
    """ Evaluates x on model, returns False if max(f(x))!=y
        and returns the runner up idx (int) otherwise

    ARGS:
        model: FFNet/Seq instance
        x: [C,H,W]  (no N)
        y: int or longtensor

    """
    pass

def linear_subindex(linear, idxs):
    # Makes a new linear model
    weight = linear.weight
    bias = linear.bias
    device = weight.device

    new_linear = nn.Linear(linear.in_features, len(idxs)).to(device)

    new_linear.weight.data.zero_()
    new_linear.bias.data.zero_()

    for i, el in enumerate(idxs):
        new_linear.weight[i].data.add_(weight[el].data)
        new_linear.bias[i].data.add_(bias[el].data)

    if hasattr(linear, 'input_shape'):
        new_linear.input_shape = linear.input_shape

    return new_linear


def replace_net(net, idxs):
    seq = list(net[:-1])
    seq.append(linear_subindex(net[-1], idxs))
    return nn.Sequential(*seq)

# =======================================================
# =           Display Utilities                         =
# =======================================================


def show_grayscale(images: torch.Tensor, size=None):
    # Tensor is a (N, 1, H, W) image with values between [0, 1.0]
    row = torch.cat([_.squeeze() for _ in images], 1).detach().cpu().numpy()

    if size is not None:
        fig, ax = plt.subplots()
        ax.grid(False)
        ax.imshow(row, cmap='gray')

    else:
        fig, ax = plt.subplots(figsize=size)
        ax.grid(False)
        ax.imshow(row, cmap='gray')


def show_rgb(images: torch.Tensor, size=None):
    # Tensor is a (N, 3, H, W) image with values between [0, 1]
    row = torch.cat([_.squeeze() for _ in images], -1).detach().cpu().numpy()
    if size is not None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=size)

    ax.grid(False)
    ax.imshow(row.transpose(1,2,0))



