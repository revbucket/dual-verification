import torch
from torchvision import transforms
import torch.nn as nn
import onnx
import onnx2pytorch
import argparse
import pickle


def convert_to_pytorch(onnx_filename):
    pytorch_model = onnx2pytorch.ConvertModel(onnx.load(onnx_filename))

    print(pytorch_model)
    modules = list(pytorch_model.modules())[1:]

    """
    Conversion steps:
    1. Collect normalization constants
    2. Collect layers
    3. modify first layer with normalization constants
    """

    layers = []
    sub_constant = 0
    div_constant = 1
    normalize = None
    subs = False
    for i, module in enumerate(modules):
        if module.__repr__().endswith('ub()'):
            sub_constant = modules[i - 1].constant
            subs = True

        if module.__repr__().endswith('iv()'):
            div_constant = modules[i- 1].constant

        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.Flatten)):
            layers.append(module)

    if isinstance(layers[-1], nn.ReLU):
        layers = layers[:-1]

    #### Handle the subtract/divide stuff by making
    if subs:
        sub_constant = sub_constant.flatten().cpu()
        div_constant = div_constant.flatten().cpu()
        normalize = transforms.Normalize(sub_constant, div_constant)

    return normalize, nn.Sequential(*layers)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input onnx file')
    parser.add_argument('--onnx', dest='onnx_filename', type=str)

    args = parser.parse_args()

    normalize, pytorch_model = convert_to_pytorch(args.onnx_filename)
    basename = args.onnx_filename.split('.')[0]
    state_dict = {'normalize': normalize, 'pytorch_model': pytorch_model}
    torch.save(state_dict, basename + '_onnx2torch.torch')


