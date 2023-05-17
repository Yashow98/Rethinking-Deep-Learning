# @Author  : Yashowhoo
# @File    : 38_timm.py
# @Description :timm is a deep-learning library created by Ross Wightman and is a collection of
# SOTA computer vision models, layers, utilities, optimizers, schedulers, data-loaders,
# augmentations and also training/validating scripts with ability to reproduce ImageNet training results.

import timm
import torch
from torch import nn
# from torchvision import models

torch.backends.cudnn.benchmark = True  # 加快训练
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model = models.resnet50()
# model = timm.create_model('resnet34', pretrained=True)
# model = timm.create_model('resnet34', num_classes=10)
model = timm.create_model('resnet50')
print(model)
# x = torch.randn(1, 3, 224, 224)
# output = model(x)
# print(output.shape)

# avail_models = timm.list_models()
# avail_pretrained_models = timm.list_pretrained()
# print(len(avail_models))  # 936
# print(avail_models[:5])
# print(len(avail_pretrained_models))  # 1163
# print(avail_pretrained_models[:5])

# It is also possible to search for model architectures using Wildcard as below:
# all_densenet_models = timm.list_models('*densenet*')
# print(all_densenet_models)
