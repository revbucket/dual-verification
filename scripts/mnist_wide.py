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
import utilities as utis

parser = argparse.ArgumentParser()
# usage: python -m scripts.mnist_ffnet 0 10 0.1
parser.add_argument('start_idx', type=int)
parser.add_argument('end_idx', type=int)
parser.add_argument('eps', type=float, default=0.1)

args = parser.parse_args()
assert args.start_idx < args.end_idx
print(args)


####################################################
PREFIX = 'exp_data/mnist_wide/'
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

    output_dict = {}
    for k, method in {'decomp_2d': eu.decomp_2d_MNIST_PARAMS,
                      'decomp_mip': decomp_2d_mip,
                      'optprox': eu.run_optprox,
                      'explp': eu.run_explp,
                      'anderson': eu.run_anderson_1cut,
                      'lp': eu.run_lp}.items():
        try:
            output_dict[k] = method(bin_net, test_input)
        except:
            print("WARNING: %s FAILED ON EXAMPLE %s" % (k, idx))


    pprint.pprint(output_dict)
    write_file(idx, output_dict)


