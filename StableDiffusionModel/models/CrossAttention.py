import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

"""
Cross attention is:
    -an attention mechanism in Transformer architecture that mixes two different embedding sequences
    -the two sequences must have the same dimension
    -the two sequences can be of different modalities (e.g. text, image, sound)
    -one of the sequences defines the output length as it plays a role of a query input
    -the other sequence then produces key and value input
    
Cross-attention Algorithm
    -Let us have embeddings (token) sequences S1 and S2
    -Calculate Key and Value from sequence S1
    -Calculate Queries from sequence S2
    -Calculate attention matrix from Keys and Queries
    -Apply queries to the attention matrix
    -Output sequence has dimension and length of sequence S2
"""


class CrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        context_dim=None,
        num_heads=8,
        debug=False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_att_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.debug = debug

        self.to_quary = nn.Linear(self.hidden_dim, self.embed_dim, bias=False)

        if self.context_dim is None:
            # Self Attention
            self.to_key = nn.Linear(self.hidden_dim, self.embed_dim, bias=False)
            self.to_value = nn.Linear(self.hidden_dim, self.embed_dim, bias=False)
        else:
            # Cross Attention
            self.to_key = nn.Linear(self.context_dim, self.embed_dim, bias=False)
            self.to_value = nn.Linear(self.context_dim, self.embed_dim, bias=False)
            self.self_attn = False

        self.out = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        )  # Not necessary

    def forward(self, tokens: torch.Tensor, context=None):
        Q = self.to_quary(tokens)
        K = self.to_key(tokens) if self.self_attn else self.to_key(context)
        V = self.to_value(tokens) if self.self_attn else self.to_value(context)

        if self.debug:
            print("Q:{}, K:{},V:{}".format(Q.shape, K.shape, V.shape))

        # transform heads onto batch dimension
        Q = rearrange(
            Q, "B T (H D) -> (B H) T D", H=self.num_att_heads, D=self.head_dim
        )
        K = rearrange(
            K, "B T (H D) -> (B H) T D", H=self.num_att_heads, D=self.head_dim
        )
        V = rearrange(
            V, "B T (H D) -> (B H) T D", H=self.num_att_heads, D=self.head_dim
        )

        if self.debug:
            print("Q:{}, K:{},V:{}".format(Q.shape, K.shape, V.shape))

        scoremats = torch.einsum("BTD,BSD->BTS", Q, K)
        attnmats = F.softmax(scoremats / math.sqrt(self.head_dim), dim=-1)
        if self.debug:
            print("scoremats:{}, attnmats:{}".format(scoremats.shape, attnmats.shape))
        ctx_vecs = torch.einsum("BTS,BSD->BTD", attnmats, V)
        # split the heads transform back to hidden.
        ctx_vecs = rearrange(
            ctx_vecs, "(B H) T D -> B T (H D)", H=self.num_heads, D=self.head_dim
        )
        return self.to_out(ctx_vecs)
