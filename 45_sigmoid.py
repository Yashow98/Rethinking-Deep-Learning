# @Author  : YashowHoo
# @File    : 45_sigmoid.py
# @Description : a highly overlooked point in the implementation of sigmoid function

import numpy as np
import torch
from torch import nn


def sigmoid(x):
    """
    numpy version

    :param x:
    :return:
    """
    if x > 0:
        # standard version for positive inputs, this prevents overflow that may occur for negative inputs.
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        # rearranged version for negative inputs, this prevents overflow that may occur for positive inputs.
        z = np.exp(x)
        return z / (1 + z)


if __name__ == '__main__':
    print(sigmoid(1000))
    print(sigmoid(-1000))
    print(sigmoid(10))
    print(sigmoid(-10))
    print(sigmoid(0))
    
    sig = nn.Sigmoid()
    print(sig(torch.tensor(1000)).item())
    print(sig(torch.tensor(-1000)).item())
    print(sig(torch.tensor(10)).item())
    print(sig(torch.tensor(-10)).item())
