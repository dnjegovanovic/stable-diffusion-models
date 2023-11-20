import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .TransformerBlock import *


class SpatialTransformer(nn.Module):
    def __init__(
        self, hidden_dim: int, context_dim: int, num_heads=8, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.norm = nn.GroupNorm(32, hidden_dim, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0
        )
        # compatibility with Diffuser, could simplify.
        self.transformer_blocks = nn.Sequential(
            TransformerBlock(hidden_dim, context_dim, num_heads)
        )
        self.proj_out = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor, cond=None, cross_attention_kwargs=None):
        b, c, h, w = x.shape
        x_in = x
        # context = rearrange(context, "b c T -> b T c")
        x = self.proj_in(self.norm(x))
        # Combine the spatial dimensions and move the channel dimen to the end
        x = rearrange(x, "b c h w->b (h w) c")
        # Apply the sequence transformer
        x = self.transformer_blocks[0](x, cond)
        # Reverse the process
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        # Residue
        return self.proj_out(x) + x_in
