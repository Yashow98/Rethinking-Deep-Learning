# @Author  : Yashowhoo
# @File    : 40_thop.py
# @Description :

import torch
from torchvision import models
from thop import profile, clever_format

model = models.resnet50()
input = torch.randn(1, 3, 224, 224)
macs, params = profile(model, inputs=(input, ))
macs, params = clever_format([macs, params])
print(macs, params)

