"""
General script structure:
1. Loads the network
2. Loops through some examples, according to the command line arguments
    a. Run optprox_all
    b. Use optprox to set up for decomp2dMIP
    c. Run explp_all(256)
    d. Use explp_all to set up for decomp2dMIP
    e. Run explp_all(512)
    f. Run explp_all(512) to set up for decomp2dMIP
    g. run LP all

"""
import time
import experiment_utils as eu
import argparse
import pickle
import pprint
import glob
import utilities as utils

import torch.optim as optim
from partitions import PartitionGroup
from dual_decompose import DecompDual
from abstract_domains import Zonotope

parser = argparse.ArgumentParser()
# usage: python -m scripts.mnist_ffnet 0 10 0.1
parser.add_argument('start_idx', type=int)
parser.add_argument('end_idx', type=int)
parser.add_argument('eps', type=float, default=0.1)

args = parser.parse_args()
assert args.start_idx < args.end_idx
print(args)



####################################################
PREFIX = 'exp_data/mnist_wide_all/'
def filenamer(idx):
    return PREFIX + str(idx) + 'lp.pkl'

def decomp_2d_mip(bin_net, test_input, preact_bounds, time_offset):
    start_time = time.time()
    get_time = lambda: time.time() - start_time + time_offset

    # Set up the decomp object
    input_shapes = [_.input_shape for _ in bin_net.net]

    partition_obj = PartitionGroup(None, style='fixed_dim', partition_rule='spatial',
                                   partition_dim=2, input_shapes=input_shapes)

    decomp = DecompDual(bin_net, test_input, Zonotope, 'partition', zero_dual=False,
                         preact_bounds=preact_bounds, partition=partition_obj)

    # Run for 2k iterations
    optim_obj = optim.Adam(decomp.parameters(), lr=2.5e-3)
    scheduler = optim.lr_scheduler.MultiplicativeLR(optim_obj, lambda epoch: 0.75, last_epoch=- 1, verbose=False)
    for i in range(5):
        last_val = decomp.dual_ascent(400, verbose=50, optim_obj=optim_obj, iter_start=i * 400)
        scheduler.step()

    # Merge for more iterations:
    decomp.merge_partitions(partition_dim={1:2, 3: 16, 5: 32, 7: 1})
    return (decomp.lagrangian().item(), get_time())



def write_file(idx, output_dict):
    with utils.safe_open(filenamer(idx), 'wb') as f:
        pickle.dump(output_dict, f)



################ MAIN SCRIPT ######################################################3



wide_net = eu.load_mnist_wide_net()

for idx in range(args.start_idx, args.end_idx):
    print('Handling MNIST example %s' % idx)


    bin_net, test_input = eu.setup_mnist_example(wide_net, idx, args.eps)


    if bin_net is None:
        print("Skipping example %s : incorrect model" % idx)
        continue

    if len(glob.glob(filenamer(idx))) >= 1:
        print("Skipping example %s: file already exists" % idx)
        continue


    # Need to be a little more careful here...

    # ===========================================================================
    # =           LP -- if this fails, continue                                 =
    # ===========================================================================
    output_dict = {}

    try:
        output_dict['lp_all'] = eu.run_lp(bin_net, test_input, use_intermed=False)
        pprint.pprint(output_dict)
        write_file(idx, output_dict)
    except Exception as err:
        print("LP FAILED ON EXAMPLE %s" % idx)
        continue


