import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .CrossAttention import CrossAttention
from .GegluModel import FeedForwardGEGLU


class TransformerBlock(nn.Module):
    """This block combine self attention and cross atention using CrossAttention block"""

    def __init__(
        self, hidden_dim: int, context_dim, num_heads=8, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.attn1 = CrossAttention(
            embed_dim=hidden_dim, hidden_dim=hidden_dim, num_heads=num_heads
        )  # self attention, we did not provvide context
        self.attn2 = CrossAttention(
            embed_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            context_dim=context_dim,
        )  # cross attention

        # self.norm_layers = nn.Sequential([nn.LayerNorm(hidden_dim) for i in range(3)])
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        # to be compatible with Diffuser, could simplify.
        # becouse we want to initialize pur network with Diffuser pretrained model
        # we need to be fully compatibile with diffuzer
        self.ff = FeedForwardGEGLU(hidden_dim)
        # standard version used in transformers.
        # self.ff = nn.Sequential(
        #     nn.Linear(hidden_dim, 3 * hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(3 * hidden_dim, hidden_dim)
        # )

    def forward(self, x: torch.Tensor, context=None, cross_attention_kwargs=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x

        return x
