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

EPS = 0.120
SKIP_VAL = -10
PREFIX = 'exp_data/eran_convSmall/'
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
        return True, torch.min(lagrange).item()
    print("Scrubbing idxs", pos_idxs)
    decomp.subindex(idxs)
    return False, torch.min(lagrange).item()


############################ MAIN SCRIPT ############################

eran_convSmall, normalize = eu.load_eran_convSmall()
if torch.cuda.is_available():
    eran_convSmall = eran_convSmall.cuda()

for idx in range(args.start_idx, args.end_idx):
    print("Handling MNIST Example %s" % idx)

    elide_net, test_input = eu.setup_mnist_example(eran_convSmall, idx, EPS, elide=True,
                                                   normalize=normalize)
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

    print(zonoinfo.box_range[-1].lbs)
    oval_range = torch.min(zonoinfo.box_range[-1].lbs).item()
    print("Example %s: OVAL MIN %s" % (idx, oval_range))
    output_dict['optprox'] = ((oval_range >= 0), time.time() - start, oval_range, idx)


    # Case when bound is too bad to even try
    if oval_range < SKIP_VAL or oval_range >= 0:
        zd_entry = (oval_range >= 0, time.time() - start, oval_range, idx)
        output_dict['ZD_premip'] = zd_entry
        output_dict['ZD_postmip'] = zd_entry

        pprint.pprint(output_dict)
        write_file(idx, output_dict)
        continue



    decomp = DecompDual(elide_net, test_input, Zonotope, choice='partition',
                        partition=PartitionGroup(None, partition_dim=2, partition_rule='similarity'),
                        preact_bounds=zonoinfo, zero_dual=True, compute_all_bounds=True,
                       lb_only=True)

    decomp.lagrangian()
    neg_idxs = (decomp.preact_bounds.box_range[-1].lbs < 0).flatten().nonzero().flatten()
    pos_idxs = (decomp.preact_bounds.box_range[-1].lbs >= 0).flatten().nonzero().flatten()
    print("Scrubbing idxs %s" % pos_idxs)
    decomp.subindex(neg_idxs)


    with torch.no_grad():
        decomp.manual_dual_ascent(2000, verbose=100, optim_params={'initial_eta': 1e-3,
                                                                   'final_eta': 1e-5,
                                                                   'betas': (0.9, 0.999)})
        passing, lagrange = scrub_idxs(decomp)
        output_dict['ZD_premip'] = (passing, time.time() - start, lagrange, idx)

        decomp.merge_partitions({5:25})
        passing, lagrange = scrub_idxs(decomp)
        if not passing:
            decomp.manual_dual_ascent(10,optim_params={'initial_eta': 1e-3,
                                                                'final_eta': 1e-4,
                                                                'betas':(0.9, 0.999)})
            passing, lagrange = scrub_idxs(decomp)
        if not passing:
            for seq in [{5:50, 3:25}, {3:50}, {5:100, 3:80}]:
                decomp.merge_partitions(seq)
                passing, lagrange = scrub_idxs(decomp)
                if passing:
                    break


    runtime = time.time() - start
    output_dict['ZD_postmip'] = (passing, runtime, lagrange, idx)
    pprint.pprint(output_dict)
    write_file(idx, output_dict)