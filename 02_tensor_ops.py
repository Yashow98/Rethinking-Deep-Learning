# @Author  : Yashowhoo
# @File    : 02_tensor_ops.py
# @Description :

import torch

a = torch.rand((2, 3))
print(a)

# mask
b = torch.zeros_like(a)
a_mask = torch.where(a > 0.5, a, b)
print(a_mask)

a_transpose = torch.transpose(a, 0, 1)
print(a_transpose)

a_unbind = torch.unbind(a, dim=1)
print(a_unbind)

b = torch.cat((a, a), dim=0)
c = torch.cat((a, a), dim=1)

print(a.shape, b.shape, c.shape)

src = torch.arange(1, 11).reshape(2, 1, 5)
print(src.shape)
print(src)

print(torch.squeeze(src))

d = torch.tensor([1, 2, 3])

# Constructs a tensor by repeating the elements of input.
# The dims argument specifies the number of repetitions in each dimension.
print(d.tile((2, )))

e = torch.tensor([[1, 2], [3, 4]])
print(torch.tile(e, (2, 2)))
print(torch.tile(e, (2, 1)))

# manual seed

torch.manual_seed(10)
