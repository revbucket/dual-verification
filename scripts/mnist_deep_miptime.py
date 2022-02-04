
import experiment_utils as eu

from collections import OrderedDict
import pickle
import time
import torch
import argparse
import copy


parser = argparse.ArgumentParser()
# usage: python -m scripts.mnist_ffnet 0 10 0.1
parser.add_argument('start_idx', type=int)
parser.add_argument('end_idx', type=int)
parser.add_argument('eps', type=float, default=0.1)

args = parser.parse_args()
assert args.start_idx < args.end_idx
print(args)




def solve_mip_partition_series(decomp_obj):
    """ Okay, rerolling with non-random partitions, but maybe different layers?"""
    output_dict = {}
    start_time = time.time()
    decomp_obj.partition.force_mip = True

    for size in [2, 4, 8]:
        start_time = time.time()
        decomp_obj.merge_partitions(partition_dim={1:2, 3:2, 5:2, 7:size, 9: 2, 11:1})
        output_dict[size] = (decomp_obj.get_ith_primal(7, decomp_obj.rhos)[0], time.time() - start_time)
        print("\t MIP DONE FOR PART OF SIZE %s IN TIME %.2f" % (size, time.time() - start_time))
    return output_dict

###############################################################################3



mnist_deep = eu.load_mnist_deep_net()


outputs = []
for idx in range(args.start_idx, args.end_idx):
    bin_net, test_input = eu.setup_mnist_example(mnist_deep, idx, 0.1, show=False)


    print('-' * 40, idx, '-' * 40)
    try:
        decomp = eu.decomp_2d_MNIST_PARAMS(bin_net, test_input, return_obj=True)
        outputs.append(solve_mip_partition_series(decomp))
    except:
        print("FAILED ON IDX %s" % idx)


    with open('mnist_deep_miptime_data_v3.pkl', 'wb') as f:
        pickle.dump(outputs, f)


