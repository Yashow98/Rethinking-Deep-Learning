# @Author  : Yashowhoo
# @File    : 04_transform.py
# @Description :

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), 1))  # one-hot
)

print(len(training_data))
img_0, label_0 = training_data[0]
print(label_0)
