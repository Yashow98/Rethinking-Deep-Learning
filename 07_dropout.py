# @Author  : Yashowhoo
# @File    : 07_dropout.py
# @Description :

import torch
from torch import nn

# 功能：随机把 Input 里面的一些元素置0，再乘以 1/1-p。
# m = nn.Dropout(p=0.2)
# a = torch.rand((2, 10))
# output = m(a)
# print(a, a.shape)
# print(output)
#
# m.eval()  # influence dropout and batch norm
# out = m(a)
# print(out)

# 功能：随机把 Input 里面某个 channel 的元素全部置0，再乘以 1/1-p。
m = nn.Dropout2d(p=0.5, inplace=True)
input = torch.randn(1, 3, 3, 3)
output = m(input)
print(input)
print(output)
