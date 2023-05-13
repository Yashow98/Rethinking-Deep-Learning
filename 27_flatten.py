# @Time    : 4/27/2023 8:33 PM
# @Author  : Yashowhoo
# @File    : 27_flatten.py
# @Description :
import torch
import torch.nn as nn

tensor = torch.randn((2, 3, 4))
output = torch.flatten(tensor)
print(output.data)
print(tensor.numel())
print(torch.numel(output))

a = torch.tensor([2.1])
b = torch.tensor([2.0])
print(torch.allclose(a, b))
