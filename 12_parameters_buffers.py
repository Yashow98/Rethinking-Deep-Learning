# @Time    : 4/12/2023 7:40 AM
# @Author  : Yashowhoo
# @File    : 12_parameters_buffers.py
# @Description :
import torch
from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        buffer = torch.randn(2, 3)  # tensor
        self.register_buffer('my_buffer', buffer)
        self.param = nn.Parameter(torch.randn(3, 3))  # 模型的成员变量

    def forward(self, x):
        # 可以通过 self.param 和 self.my_buffer 访问
        pass


model = MyModel()
for param in model.parameters():
    print(param)
print("----------------")
for buffer in model.buffers():
    print(buffer)
print("----------------")
print(model.state_dict())
