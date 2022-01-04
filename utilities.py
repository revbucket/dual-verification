""" Utilities that are useful in general """
import torch
import matplotlib.pyplot as plt
import random
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




