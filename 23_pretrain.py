# @Time    : 4/18/2023 9:00 PM
# @Author  : Yashowhoo
# @File    : 23_pretrain.py
# @Description :
import torch
import torch.nn as nn
from torchvision import models
from torchinfo import summary

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# net = models.resnet18(pretrained=True).to(device=device)
# in_channel = net.fc.in_features
# net.fc = nn.Linear(in_channel, 6)

net = models.alexnet().to(device)
# summary(net, input_size=(1, 3, 224, 224))
net.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(256 * 6 * 6, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, 6),
)

# 初始化权重
for m in net.classifier.modules():
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

for p in net.classifier.parameters():
    print(p)

# option2
# net = resnet34(num_classes=5)
# pre_weights = torch.load(model_weight_path, map_location=device)
# del_key = []
# for key, _ in pre_weights.items():
#     if "fc" in key:
#         del_key.append(key)
#
# for key in del_key:
#     del pre_weights[key]
#
# missing_keys, unexpected_keys = net.load_state_dict(pre_weights, strict=False)
# print("[missing_keys]:", *missing_keys, sep="\n")
# print("[unexpected_keys]:", *unexpected_keys, sep="\n")

# pre_weights = torch.load(model_weight_path, map_location='cpu')
#
# # delete classifier weights
# pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
# missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)
# freeze features weights
# for param in net.features.parameters():
#     param.requires_grad = False

# for name, para in net.named_parameters():
#     # 除最后的全连接层外，其他权重全部冻结
#     if "fc" not in name:
#         para.requires_grad_(False)

# pg = [p for p in model.parameters() if p.requires_grad]
# optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=4E-5)

# lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
# scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
