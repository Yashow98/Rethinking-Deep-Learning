# @Time    : 5/4/2023 10:01 PM
# @Author  : Yashowhoo
# @File    : 35_position_encoding.py
# @Description :4种position embedding
import torch
import torch.nn as nn

# 1d absolute sin cos constant embedding
# Transformer position embedding
def create_1d_absolute_sincos_embedding(n_pos_vec, dim):
    # n_pos_vec = torch.arange(n_pos)  # index

    assert dim % 2 == 0, "wrong dimension"
    position_embedding = torch.zeros(n_pos_vec.numel(), dim, dtype=torch.float)

    omega = torch.arange(dim//2, dtype=torch.float)  # refer to i
    omega /= dim / 2.
    omega = 1. / (10000 ** omega)

    out = n_pos_vec[:, None] @ omega[None, :]  # shape is n_pos_vec * dim/2

    embsin = torch.sin(out)
    embcos = torch.cos(out)

    position_embedding[:, 0::2] = embsin
    position_embedding[:, 1::2] = embcos

    return position_embedding

# vit model position embedding
def create_1d_absolute_trainable_embeddings(n_pos_vec, dim):
    # n_pos_vec = num of channels or patches
    position_embedding = nn.Embedding(num_embeddings=n_pos_vec.numel(), embedding_dim=dim, dtype=torch.float)
    nn.init.constant_(position_embedding.weight, 0.)

    return position_embedding


if __name__ == '__main__':
    n_pos = 10
    d_model = 20

    pos_vec = torch.arange(n_pos, dtype=torch.float)
    pos_emb = create_1d_absolute_sincos_embedding(n_pos_vec=pos_vec, dim=d_model)
    print(pos_emb)
    print(pos_emb.shape)

