# @Author  : Yashowhoo
# @File    : 03_dataset_loader.py
# @Description :

import os
import pandas as pd
from torchvision.io import read_image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(root='data', train=True, transform=ToTensor(), download=False)
test_data = datasets.FashionMNIST(root='data', train=False, transform=ToTensor(), download=False)

print(len(training_data), len(test_data))

# img_0, label_0 = training_data[0]
# print(img_0.shape, label_0)
#
# img_0 = torch.squeeze(img_0)
# plt.imshow(img_0, cmap='gray')
# plt.show()

# The labels.csv file looks like:
# tshirt1.jpg, 0
# tshirt2.jpg, 0
# ......
# ankleboot999.jpg, 9


# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def __len__(self):
#         return len(self.img_labels)
#
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#
#         # 图片预处理函数
#         if self.transform:
#             image = self.transform(image)
#
#         # 标签预处理函数
#         if self.target_transform:
#             label = self.target_transform(label)
#
#         return image, label


# 数据的打包，构建mini_batch，得到一个迭代器
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Display image and label.
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")

print(len(train_dataloader))  # return num of batch
