# @Time    : 5/13/2023 7:04 PM
# @Author  : Yashowhoo
# @File    : 39_loss.py
# @Description :
import torch
from torch import nn

# This criterion computes the cross entropy loss between input logits and target.
# ce_loss = nn.CrossEntropyLoss(reduction="none")
# logits = torch.randn((3, 5), requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)  # delta目标分布
# target = torch.randn((3, 5)).softmax(dim=1)  # 内部都是非delta目标分布实现
# target = torch.randint(5, size=(3, ))
# print(target)

# ce_output = ce_loss(logits, target)
# print(ce_output)
# output.backward()

# The negative log likelihood loss.
# m = nn.LogSoftmax(dim=1)
# logits = m(logits)
nll_loss = nn.NLLLoss()
# nll_output = nll_loss(logits, target)
#
# print(nll_output)

# print(torch.allclose(ce_output, nll_output))  # True

# The Kullback-Leibler divergence loss.
# kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
# kl_loss = nn.KLDivLoss(reduction="none")
# kl_output = kl_loss(logits, target).sum(dim=1)
# print(kl_output)

# ce = ie + kl
# H(p, q) = H(p) + KL(p || q)

# target_entropy = torch.distributions.Categorical(probs=target).entropy()  # 默认为常数，如果是delta分布，值等于0
# print(target_entropy)

# output = kl_output + target_entropy
# print(torch.allclose(output, ce_output))  # True

# Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities:
bce_loss = nn.BCELoss()
bce_logits = nn.BCEWithLogitsLoss()

input = torch.randn(3, requires_grad=True)
prob_1 = torch.sigmoid(input)
prob_0 = 1 - prob_1.unsqueeze_(-1)
# print(prob_0)
# print(prob_1)
prob = torch.cat((prob_0, prob_1), dim=-1)
# print(prob)
target = torch.empty(3).random_(2)

bce_output = bce_loss(torch.sigmoid(input), target)
bce_logits_output = bce_logits(input, target)

bce_nl = nll_loss(torch.log(prob), target.long())
print(bce_output)
print(bce_logits_output)
print(bce_nl)
print(torch.allclose(bce_nl, bce_output))

# cosine similarity loss
# input1 = torch.randn(4, 512)
# input2 = torch.randn(4, 512)
# target = torch.randint(2, size=(4, )) * 2 - 1
# cos_loss = nn.CosineEmbeddingLoss()
# cos_output = cos_loss(input1, input2, target)
# print(cos_output)
