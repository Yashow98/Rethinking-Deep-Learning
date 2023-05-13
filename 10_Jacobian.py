# @Time    : 4/11/2023 4:29 PM
# @Author  : Yashowhoo
# @File    : 10_Jacobian.py
# @Description :
import torch
from torch.autograd.functional import jacobian

def exp_reducer(x):
    return x.exp().sum(dim=1)


inputs = torch.rand((2, 2), requires_grad=True)
output = exp_reducer(inputs)
print(output)

jacb = jacobian(exp_reducer, inputs)
print(jacb)

output.backward(torch.ones_like(output))  # 张量对张量偏导
print(inputs.grad)

print(torch.ones_like(output) @ jacb  )

a = torch.rand((2, 3), requires_grad=True)
b = torch.rand((3, 4), requires_grad=True)
y = a @ b

y.backward(torch.ones_like(y))
print(a.grad)
print(b.grad)
