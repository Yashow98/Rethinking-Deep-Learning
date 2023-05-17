# @Author  : Yashowhoo
# @File    : 28_transposeconv.py
# @Description :

import torch
import torch.nn as nn

# unfold = nn.Unfold(kernel_size=(2, 3))
# a = torch.randn((2, 5, 3, 4))
# output = unfold(a)  # 滑动窗口4个blocks, 5个channels
# print(output.size())
# a = torch.randn((2, 3))
# a_transpose = a.transpose_(0, 1)
# print(a_transpose)
# print(a)

# input_feature_map = torch.randn((1, 3, 16, 16))
# downsample = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)
# output_fearture_map = downsample(input_feature_map)
# print(output_fearture_map.size())
#
# upsample = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)
# output = upsample(output_fearture_map, output_size=input_feature_map.size())
# print(output.size())

a = torch.empty((3, 3, 4, 2))
# pad = (1, 1)  # pad only the last dim
# out = nn.functional.pad(a, pad, value=0)
# print(out.size())

# out = nn.functional.pad(a, pad=(1, 1, 1, 1), value=0)  # pad the last 2 dim
# print(out.size())

out = nn.functional.pad(a, pad=(1, 1, 1, 1, 2, 2), value=1)  # pad the last 3 dim
print(out.size())
