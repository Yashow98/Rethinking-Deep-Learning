# @Author  : Yashowhoo
# @File    : 42_cosine_similarity.py
# @Description :

import numpy as np
from numpy import linalg
import torch


a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
eps = 1e-08

cosine_similarity = np.dot(a, b) / max(linalg.norm(a) * linalg.norm(b), eps)
print(cosine_similarity)  # 0.9925833339709303

a = torch.tensor([1, 2, 3], dtype=torch.float)
b = torch.tensor([2, 3, 4], dtype=torch.float)

cosine_sim = torch.cosine_similarity(a, b, dim=0)
print(cosine_sim.item())  # 0.9925832748413086
