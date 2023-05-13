# @Time    : 4/11/2023 3:08 PM
# @Author  : Yashowhoo
# @File    : 08_nnseq.py
# @Description :
import torch
from torch import nn

# 都是继承Module
model = nn.Sequential(
    nn.Linear(3, 4),
    nn.Linear(4, 10),
)

print(dir(model))
print(model)
