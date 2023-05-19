# @Author  : YashowHoo
# @File    : 44_gather.py
# @Description : torch.gather() function is used to create a new tensor
# by gathering elements from an input tensor along a specific dimension and from specific indices.

import torch

# creating 1-d tensor
tensor1 = torch.arange(10) * 2
print(tensor1)
print(tensor1.shape)

output1 = torch.gather(tensor1, dim=0, index=torch.tensor([0, 1, 2]))
print(output1)

# creating 2-d tensor
tensor2 = torch.arange(9).reshape(3, 3)
print(tensor2)
output2 = torch.gather(tensor2, dim=0, index=torch.tensor([[0, 1, 2],
                                                           [1, 2, 0]]))
print(output2)

output3 = torch.gather(tensor2, dim=1, index=torch.tensor([[0, 1, 2],
                                                           [1, 2, 0]]))
print(output3)
