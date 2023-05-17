# @Author  : Yashowhoo
# @File    : 09_autograd.py
# @Description :

import torch
from torch.nn import functional

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")
# print(x.requires_grad)
# x.requires_grad_(True)
# print(x.requires_grad)

loss.backward()
print(w.grad)
print(b.grad)

z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

# Another way to achieve the same result is to use the detach() method on the tensor:
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

# 标量对矩阵求偏导，计算一个雅可比矩阵，而张量对张量求偏导，会计算一个雅可比矩阵与张量的点积
# Notice that when we call backward for the second time with the same argument,
# the value of the gradient is different. This happens because when doing backward propagation,
# PyTorch accumulates the gradients, i.e.
# the value of computed gradients is added to the grad property of all leaf nodes of computational graph.
# If you want to compute the proper gradients,
# you need to zero out the grad property before. In real-life training an optimizer helps us to do this.
inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")
