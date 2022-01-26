import time
import experiment_utils as eu
import argparse
import pickle
import pprint
import glob
import utilities as utils

import torch.optim as optim
from partitions import PartitionGroup
from dual_decompose2 import DecompDual2
from abstract_domains import Zonotope, Hyperbox
from neural_nets import BoxInformedZonos

parser = argparse.ArgumentParser()
# usage: python -m scripts.mnist_ffnet 0 10 0.1
parser.add_argument('start_idx', type=int)
parser.add_argument('end_idx', type=int)
parser.add_argument('eps', type=float, default=0.1)

args = parser.parse_args()
assert args.start_idx < args.end_idx
print(args)



######################

PREFIX = 'exp_data/mnist_ffnet_ablate/'
def filenamer(idx):
    return PREFIX + str(idx) + '.pkl'

def write_file(idx, output_dict):
    with utils.safe_open(filenamer(idx), 'wb') as f:
        pickle.dump(output_dict, f)



def ablate(bin_net, test_input, use_optprox=False):
    output = {}
    suffix = ''
    start_time = time.time()
    get_time = lambda: time.time() - start_time
    preact_bounds = None
    if use_optprox:
        suffix = '_optprox'
        optprox_all = eu.run_optprox(bin_net, test_input, use_intermed=False, return_model=True)[0]
        preact_bounds = eu.ovalnet_to_zonoinfo(optprox_all, bin_net, test_input)
    else:
        lbs, ubs = eu.get_best_naive_kw(bin_net, test_input)
        hboxes = [Hyperbox(lb.flatten(), ub.flatten()) for (lb, ub) in zip(lbs, ubs)]
        preact_bounds = BoxInformedZonos(bin_net, test_input, box_range=hboxes).compute()

    partition_obj = PartitionGroup(None, style='fixed_dim', partition_rule='similarity',
                                   partition_dim=2)
    decomp = DecompDual2(bin_net, test_input, Zonotope, 'partition', zero_dual=False,
                         partition=partition_obj, preact_bounds=preact_bounds)

    optim_obj = optim.Adam(decomp.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.MultiplicativeLR(optim_obj, lambda epoch: 0.75, last_epoch=- 1, verbose=False)
    for i in range(10):
        last_val = decomp.dual_ascent(100, verbose=200, optim_obj=optim_obj, iter_start=i * 100)
        scheduler.step()

    output['decomp2d%s' % suffix] = (last_val.item() , get_time())

    decomp.merge_partitions(partition_dim={1:2, 3:2, 5: 2, 7: 32, 9:1})
    output['decomp2d_smallMIP%s' % suffix] = (decomp.lagrangian().item(), get_time())

    optim_obj = optim.Adam(decomp.parameters(), lr=5e-3)
    scheduler = optim.lr_scheduler.MultiplicativeLR(optim_obj, lambda epoch: 0.2, last_epoch=- 1, verbose=False)
    for i in range(2):
        last_val = decomp.dual_ascent(25, verbose=True, optim_obj=optim_obj, iter_start=i * 25)
        scheduler.step()
    output['decomp2d_MIPascent%s' % suffix] = (last_val.item(), get_time())

    decomp.merge_partitions(partition_dim={1:16, 3:16,5:32, 7: 32, 9: 1})
    output['decomp2d_bigMIP%s' % suffix] = (decomp.lagrangian().item(), get_time())
    return output



################################### MAIN SCRIPT #########################3

ffnet = eu.load_mnist_ffnet()

for idx in range(args.start_idx, args.end_idx):
    print('Handling MNIST example %s' % idx)


    bin_net, test_input = eu.setup_mnist_example(ffnet, idx, args.eps)


    if bin_net is None:
        print("Skipping example %s : incorrect model" % idx)
        continue

    if len(glob.glob(filenamer(idx))) >= 1:
        print("Skipping example %s: file already exists" % idx)
        continue

    output_dict = {}


    # RUN LP
    output_dict['lp'] = eu.run_lp(bin_net, test_input, use_intermed=False)


    # Now run Optprox and optprox all

    output_dict['optprox'] = eu.run_optprox(bin_net, test_input, use_intermed=True)
    output_dict['optprox_all'] = eu.run_optprox(bin_net, test_input, use_intermed=False)

    # And then run without the optprox bounds
    output_dict.update(ablate(bin_net, test_input, use_optprox=False))
    output_dict.update(ablate(bin_net, test_input, use_optprox=True))


    pprint.pprint(output_dict)
    write_file(idx, output_dict)