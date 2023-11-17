import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .CrossAttention import CrossAttention


class TransformerBlock(nn.Module):
    def __init__(
        self, hidden_dim: int, context_dim, num_heads=8, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.attn_1 = CrossAttention(
            embed_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        ) # self attention, we did not provide context_dim

        self.attn_2 = CrossAttention(
            embed_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            context_dim=context_dim
        ) # cross attention
        
        self.norm_1 = nn.LayerNorm(hidden_dim)
        self.norm_2 = nn.LayerNorm(hidden_dim)
        self.norm_3 = nn.LayerNorm(hidden_dim)
        
        # standard version used in transformers.
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 3 * hidden_dim),
            nn.GELU(),
            nn.Linear(3 * hidden_dim, hidden_dim)
        )
    
    def forward(self, x:torch.Tensor, context=None):
        x = self.attn_1(self.norm_1(x)) + x
        x = self.attn_2(self.norm_2(x), context) + x
        x = self.ff(self.norm_3(x)) + x
        
        return x