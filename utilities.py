""" Utilities that are useful in general """
import torch
import matplotlib.pyplot as plt
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




