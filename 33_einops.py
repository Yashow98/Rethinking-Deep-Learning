# @Author  : Yashowhoo
# @File    : 33_einops.py
# @Description :Flexible and powerful tensor operations for readable and reliable code.
# Supports numpy, pytorch, tensorflow, jax, and others.

import torch
from einops import rearrange, reduce, repeat

input_tensor = torch.randn((4, 3, 7, 7))
out1 = rearrange(input_tensor, 'b c h w -> b h c w')
out2 = input_tensor.transpose(1, 2)
print(out1.shape, out2.shape)
print(torch.allclose(out1, out2))

# composing axes
out3 = rearrange(input_tensor, 'b c h w -> (b c) h w')
out4 = input_tensor.reshape(12, 7, 7)
print(out3.shape, out4.shape)
print(torch.allclose(out3, out4))

# decomposing axes
out5 = rearrange(out3, '(b c) h w -> b c h w', b=4)
out6 = out3.reshape(4, 3, 7, 7)
print(out6.shape, out5.shape)
print(torch.allclose(out5, out6))

# reduce

# average over batch
mean = reduce(input_tensor, 'b c h w -> c h w', 'mean')
torch_mean = torch.mean(input_tensor, dim=0)
print(mean.shape, torch_mean.shape)
print(torch.allclose(mean, torch_mean))

# repeating elements
add_axes = repeat(mean, 'c h w -> 4 c h w')
tile = torch.tile(mean, dims=(4, 1, 1, 1))
print(add_axes.shape, tile.shape)
print(torch.allclose(add_axes, tile))
