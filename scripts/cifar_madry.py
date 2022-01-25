"""
General script structure:
1. Loads the network
2. Loops through some examples, according to the command line arguments
    a. Run optprox
    b. Run explp
    c. Run anderson1cut
    d. Run lp
    e. Run decomp2d
    f. Run decomp2d_mip
    LAST LAYER ONLY FOR ALL THE THINGS^^^^

    g. Save these to a file, based on the example id (making LOTS of examples)

"""
import time
import experiment_utils as eu
import argparse
import pickle
import pprint

parser = argparse.ArgumentParser()
# usage: python -m scripts.mnist_ffnet 0 10 0.1
parser.add_argument('start_idx', type=int)
parser.add_argument('end_idx', type=int)
parser.add_argument('--eps', type=float, default=0.04705882352, required=False)

args = parser.parse_args()
assert args.start_idx < args.end_idx
print(args)


####################################################

def decomp_2d_mip(bin_net, test_input):
    start_time = time.time()
    decomp_obj= eu.decomp_2d_CIFAR_PARAMS(bin_net, test_input, return_obj=True)
    decomp_obj.merge_partitions(partition_dim={1:2, 3:2, 5:20, 7:1,})
    return decomp_obj.lagrangian().item(), time.time() - start_time


def write_file(idx, output_dict):
    PREFIX = 'exp_data/cifar_madry_'
    with open(PREFIX + str(idx) + '.pkl', 'wb') as f:
        pickle.dump(output_dict, f)



################ MAIN SCRIPT ######################################################3



cifar_madry = eu.load_cifar_madry()

for idx in range(args.start_idx, args.end_idx):
    print('Handling CIFAR example %s' % idx)


    bin_net, test_input = eu.setup_cifar_example(cifar_madry, idx, args.eps)


    if bin_net is None:
        print("Skipping example %s : incorrect model" % idx)
        continue


    output_dict = {}
    output_dict['decomp_2d'] = eu.decomp_2d_CIFAR_PARAMS(bin_net, test_input)
    output_dict['decomp_mip'] = decomp_2d_mip(bin_net, test_input)

    output_dict['optprox'] = eu.run_optprox(bin_net, test_input, use_intermed=True)
    output_dict['explp'] = eu.run_explp(bin_net, test_input, use_intermed=True)
    output_dict['anderson'] = eu.run_anderson_1cut(bin_net, test_input, use_intermed=True)
    output_dict['lp'] = eu.run_lp(bin_net, test_input, use_intermed=True)



    pprint.pprint(output_dict)
    write_file(idx, output_dict)


