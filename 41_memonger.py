# @Time    : 5/14/2023 10:08 AM
# @Author  : Yashowhoo
# @File    : 41_memonger.py
# @Description :By replacing nn.Sequential with memonger.SublinearSequential,
# the memory required for backward is reduced from O(N) to O(sqrt(N)).
import torch
from torch import nn
import memonger

if __name__ == '__main__':
    data = torch.randn(1, 3, 5, 5)

    net = memonger.SublinearSequential(
        nn.Conv2d(3, 3, kernel_size=3, padding=1),
        nn.BatchNorm2d(3),
        nn.ReLU(),
        nn.Conv2d(3, 3, kernel_size=3, padding=1),
        nn.BatchNorm2d(3),
        nn.ReLU(),
        nn.Conv2d(3, 3, kernel_size=3, padding=1),
        nn.BatchNorm2d(3),
        nn.ReLU(),
    )
    # output = net(data)
    # print(output.shape)
    # res1 = net(data).sum()
    #
    net.set_reforward(False)
    output = net(data)
    print(output.shape)
    # res2 = net(data).sum()
    #
    # net2 = nn.Sequential(
    #     *list(net.children())
    # )
    # output = net2(data)
    # print(output.shape)
    # res3 = net2(data).sum()
    #
    # print(res1.data, res2.data, res3.data)
