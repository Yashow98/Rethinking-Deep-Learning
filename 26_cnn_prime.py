# @Time    : 4/27/2023 4:49 PM
# @Author  : Yashowhoo
# @File    : 26_cnn_prime.py
# @Description :重新思考卷积
import torch
import torch.nn as nn
import torch.nn.functional as F

conv_layer = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, bias=True)
input_feature_map = torch.randn((1, 3, 4, 4))
output_feature_map = conv_layer(input_feature_map)

print(conv_layer)
print(conv_layer.weight.data)
print(conv_layer.bias.data)

print(input_feature_map)
print(output_feature_map)

