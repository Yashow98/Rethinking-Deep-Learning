# @Author  : Yashowhoo
# @File    : 01_tensor.py
# @Description :

import torch
import numpy as np

print(torch.__version__)
print(torch.cuda.is_available())

a = torch.tensor([1, 2, 3])
print(a)
print(a.dtype)

b = torch.tensor([1., 2])
print(b.dtype)

c = np.random.rand(3, 2)
print(c)

d = torch.tensor(c)
print(d)

e = torch.ones_like(d)
print(e)

f = torch.rand((3, 2))
print(f)
print(f.shape)
print(f.device)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
g = f.to(device)
print(g)

print(torch.is_tensor(g))

print(torch.numel(a))

h = torch.from_numpy(c)
print(h)

print(torch.full_like(h, 4))

