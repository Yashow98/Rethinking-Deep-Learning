# @Author  : Yashowhoo
# @File    : 41_memonger.py
# @Description :By replacing nn.Sequential with memonger.SublinearSequential,
# the memory required for backward is reduced from O(N) to O(sqrt(N)).

import torch
from torch import nn
import memonger

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    net = net.to(device)
    print(f"{torch.cuda.memory_allocated() / (1024 * 1024)} M, {torch.cuda.memory_allocated()} B")  # 0.01025390625 M, 10752 B

    # output = net(data)
    # res1 = net(data).sum()
    #
    # net.set_reforward(False)
    # output = net(data)
    # res2 = net(data).sum()
    #
    # net2 = nn.Sequential(
    #     *list(net.children())
    # )
    # output = net2(data)
    # res3 = net2(data).sum()
    # print(res1.data, res2.data, res3.data)



