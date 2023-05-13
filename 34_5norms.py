# @Time    : 5/4/2023 8:38 PM
# @Author  : Yashowhoo
# @File    : 34_5norms.py
# @Description :
import torch
import torch.nn as nn

batch_size = 2
time_step = 3
embedding_dim = 4
numgroups = 2
inputx = torch.randn((batch_size, time_step, embedding_dim))  # N * L * C
# layernorm
# layernorm = nn.LayerNorm(embedding_dim, elementwise_affine=False)
# outputy = layernorm(inputx)
# print(outputy.shape)
#
# input_mean = inputx.mean(dim=-1, keepdim=True)
# input_std = inputx.std(dim=-1, keepdim=True, unbiased=False)
# verify_y = (inputx - input_mean) / (input_std + 1e-5)
# print(verify_y.shape)
# print(torch.allclose(outputy, verify_y))

# batchnorm
# batchnorm = nn.BatchNorm1d(embedding_dim, affine=False)
# batchnorm_op = batchnorm(inputx.transpose(-1, -2)).transpose(-1, -2)
#
# input_mean = inputx.mean(dim=(0, 1), keepdim=True)
# input_std = inputx.std(dim=(0, 1), keepdim=True, unbiased=False)
# verify_batchnorm = (inputx - input_mean) / (input_std + 1e-5)
#
# print(verify_batchnorm.shape)
# print(torch.allclose(batchnorm_op, verify_batchnorm))

# instancenorm
# insnorm = nn.InstanceNorm1d(embedding_dim)
# insnorm_op = insnorm(inputx.transpose(-1, -2)).transpose(-1, -2)
#
# input_mean = inputx.mean(dim=1, keepdim=True)
# input_std = inputx.std(dim=1, keepdim=True, unbiased=False)
# verify_insnorm = (inputx - input_mean) / (input_std + 1e-5)
# print(insnorm_op)
# print(verify_insnorm)

# groupnorm
groupnorm = nn.GroupNorm(num_groups=numgroups, num_channels=embedding_dim, affine=False)
groupnorm_op = groupnorm(inputx.transpose(-1, -2)).transpose(-1, -2)

group_input = torch.split(inputx, split_size_or_sections=embedding_dim//numgroups, dim=-1)  # return tuple
group_res = []
for each_group in group_input:
    group_mean = each_group.mean(dim=(1, 2), keepdim=True)
    group_std = each_group.std(dim=(1, 2), keepdim=True, unbiased=False)
    each_group_norm = (each_group - group_mean) / (group_std + 1e-5)
    group_res.append(each_group_norm)

verify_groupnorm = torch.cat(group_res, dim=-1)
# print(groupnorm_op)
# print(verify_groupnorm)
print(torch.allclose(groupnorm_op, verify_groupnorm))


