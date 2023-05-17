# @Author  : Yashowhoo
# @File    : 32_cudnn.py
# @Description :

import torch

print(torch.backends.cudnn.version())
print(torch.backends.cudnn.is_available())

# A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
print(torch.backends.cudnn.benchmark)

