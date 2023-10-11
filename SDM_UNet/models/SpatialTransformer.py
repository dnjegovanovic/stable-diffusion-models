import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from TransformerBlock import *


class SpatialTransformer(nn.Module):
    def __init__(
        self, hidden_dim: int, context_dim: int, num_heads=8, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.norm_layer = nn.GroupNorm(32, hidden_dim, eps=1e-6, affine=True)
        self.projection_input = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0
        )
        # compatibility with Diffuser, could simplify.
        self.transformer_block = nn.Sequential(
            TransformerBlock(hidden_dim, context_dim, num_heads)
        )
        self.projection_output = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor, cond=None):
        b, c, h, w = x.shape
        x_in_tmp = x
        x = self.projection_input(self.norm_layer(c))
        x = rearrange(x, "b c h w->b (h w) c")
        x = self.transformer_block[0](x, cond)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        return self.projection_output(x) + x_in_tmp
