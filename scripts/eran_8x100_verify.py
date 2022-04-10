""" General script structure:
1. Loads the network
2. Loop through examples, according to command line args
3. For each example, do:
    a. Run optprox all

    ---- skip if logit diff > 10
    b. make decomp object
    c. cut out any indices of positive LBs
    d. run dual ascent
    e. cut out any indices of positive LBs
    f. progressively tighten MIPS
        i) if ever positive, remove those idxs

    For example with idx i, we pickle the following dict to file
    {
     optprox: (true/false, runtime (s), i)
     ZD     : (true/false, runtime (s), i)
    }
    Where true/false indicates whether or not we can verify with the key method
"""

import time
import experiment_utils as eu
import argparse
import pickle
import pprint
import glob
import utilities as utils
import torch
from dual_decompose import DecompDual
from abstract_domains import Hyperbox, Zonotope
from partitions import PartitionGroup

parser = argparse.ArgumentParser()
# usage: python -m scripts.mnist_ffnet 0 10 0.1
parser.add_argument('start_idx', type=int)
parser.add_argument('end_idx', type=int)
#parser.add_argument('eps', type=float, required=False, default=0.026)

args = parser.parse_args()


assert args.start_idx < args.end_idx
print(args)

EPS = 0.026
SKIP_VAL = -10
PREFIX = 'exp_data/eran_8x100/'
def filenamer(idx):
    return PREFIX + str(idx) + '.pkl'

def decomp_2d_mip(bin_net, test_input):
    start_time = time.time()
    decomp_obj= eu.decomp_2d_MNIST_PARAMS(bin_net, test_input, return_obj=True)
    decomp_obj.merge_partitions(partition_dim={1:2, 3:2, 5:20, 7: 1})
    return decomp_obj.lagrangian().item(), time.time() - start_time


def write_file(idx, output_dict):
    with utils.safe_open(filenamer(idx), 'wb') as f:
        pickle.dump(output_dict, f)


def scrub_idxs(decomp):
    lagrange = decomp.lagrangian()
    print(lagrange)
    idxs = (lagrange < 0).flatten().nonzero().flatten()
    pos_idxs = (lagrange >= 0).flatten().nonzero().flatten()
    if len(idxs) == 0:
        return True
    print("Scrubbing idxs", pos_idxs)
    decomp.subindex(idxs)
    return False


############################ MAIN SCRIPT ############################

eran_8x100 = eu.load_eran_8x100()
if torch.cuda.is_available():
    eran_8x100 = eran_8x100.cuda()

for idx in range(args.start_idx, args.end_idx):
    print("Handling MNIST Example %s" % idx)

    elide_net, test_input = eu.setup_mnist_example(eran_8x100, idx, EPS, elide=True)
    print(elide_net)
    if elide_net is None:
        print("Skipping example %s : incorrect model " % idx)
    if len(glob.glob(filenamer(idx))) >= 1:
        print("Skipping example %s: file already exists" % idx)
        continue

    start = time.time()
    output_dict = {}
    ovalout = eu.run_optprox(elide_net, test_input, use_intermed=False, return_model=True)
    zonoinfo = eu.ovalnet_to_zonoinfo(ovalout[0], elide_net, test_input)

    print(zonoinfo.box_range)
    oval_range = torch.min(zonoinfo.box_range[-1].lbs)
    print("Example %s: OVAL MIN %s" % (idx, oval_range))
    output_dict['optprox'] = ((oval_range >= 0).item(), time.time() - start, idx)

    if oval_range < SKIP_VAL:
        output_dict[idx] = (False, time.time() - start)
        pprint.pprint(output_dict)
        write_file(idx, output_dict)
        continue




    decomp = DecompDual(elide_net, test_input, Zonotope, choice='partition',
                        partition=PartitionGroup(None, partition_dim=2, partition_rule='similarity'),
                        preact_bounds=zonoinfo, zero_dual=False, compute_all_bounds=True,
                       lb_only=True)

    decomp.lagrangian()
    neg_idxs = (decomp.preact_bounds.box_range[-1].lbs < 0).flatten().nonzero().flatten()
    pos_idxs = (decomp.preact_bounds.box_range[-1].lbs >= 0).flatten().nonzero().flatten()
    print("Scrubbing idxs %s" % pos_idxs)
    decomp.subindex(neg_idxs)


    #idxs = (decomp.preact_bounds.box_range[-1].lbs < 0).nonzero().flatten()
    with torch.no_grad():
        decomp.manual_dual_ascent(1000, verbose=100)
        scrub_idxs(decomp)
        passing = False
        for seq in [{15:25, 13:25, 11:25}, {9:25, 7:25}, {5:25, 3:25, 1:25}, {9:50, 7:50}, {5:50, 3:50}]:
            decomp.merge_partitions(seq)
            passing = scrub_idxs(decomp)
            if passing:
                break
    if not passing:
        # Final shot: try to solve the full MIPs with time limits
        pass

    runtime = time.time() - start
    output_dict['ZD'] = (passing, runtime, idx)
    pprint.pprint(output_dict)
    write_file(idx, output_dict)