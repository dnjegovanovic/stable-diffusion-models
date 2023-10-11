import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GEGLUproj(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.proj = nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForwardGEGLU(nn.Module):
    # https://github.com/huggingface/diffusers/blob/95414bd6bf9bb34a312a7c55f10ba9b379f33890/src/diffusers/models/attention.py#L339
    # A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.
    def __init__(self, hidden_dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            GEGLUproj(hidden_dim, mult * hidden_dim),
            nn.Dropout(0.0),
            nn.Linear(mult * hidden_dim, hidden_dim),
        )  # to be compatible with Diffuser, could simplify.

    def forward(self, x: torch.Tensor):
        return self.net(x)
