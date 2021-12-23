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



# ========================================================
# =           Other helpful methods                      =
# ========================================================

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




