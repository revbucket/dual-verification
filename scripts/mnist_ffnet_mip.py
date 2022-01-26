import time
import experiment_utils as eu
import argparse
import pickle
import pprint
import glob

from mip_verify import MIPVerify
from neural_nets import PreactBounds
from abstract_domains import Zonotope
import utilities as utils

parser = argparse.ArgumentParser()
# usage: python -m scripts.mnist_ffnet 0 10 0.1
parser.add_argument('start_idx', type=int)
parser.add_argument('end_idx', type=int)
parser.add_argument('eps', type=float, default=0.1)

args = parser.parse_args()
assert args.start_idx < args.end_idx
print(args)


PREFIX = 'exp_data/mnist_ffnet_all/'
def filenamer(idx):
    return PREFIX + str(idx) + 'mip.pkl'

def write_file(idx, output_dict):
    with utils.safe_open(filenamer(idx), 'wb') as f:
        pickle.dump(output_dict, f)


def full_mipverify(bin_net, test_input, gurobi_params=None):
    start_time = time.time()
    preact_bounds = PreactBounds(bin_net, test_input, Zonotope).compute()

    mipverify_obj = MIPVerify(bin_net, test_input, preact_bounds)
    bound = mipverify_obj.compute(verbose=True, gurobi_params=gurobi_params)
    return bound, time.time() - start_time



ffnet = eu.load_mnist_ffnet()

for idx in range(args.start_idx, args.end_idx):
    print('Handling MNIST example %s' % idx)


    bin_net, test_input = eu.setup_mnist_example(ffnet, idx, args.eps)

    bin_net = bin_net.cpu()
    ffnet = ffnet.cpu()


    if bin_net is None:
        print("Skipping example %s : incorrect model" % idx)
        continue

    if len(glob.glob(filenamer(idx))) >= 1:
        print("Skipping example %s: file already exists" % idx)
        continue

    output_dict = {'mip': full_mipverify(bin_net, test_input)}
    write_file(idx, output_dict)