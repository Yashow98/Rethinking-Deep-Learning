# @Time    : 4/17/2023 4:09 PM
# @Author  : Yashowhoo
# @File    : 21_cudamem.py
# @Description :
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

a = torch.rand((1024, 1024), dtype=torch.float32).to(device)  # 4M
print(f'torch mem {torch.cuda.memory_allocated() / (1024 * 1024)} M, {torch.cuda.memory_allocated()} B')

