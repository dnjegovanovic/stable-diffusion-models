import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class CrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
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
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.debug = debug
        self.query = nn.Linear(hidden_dim, embed_dim, bias=False)
        if context_dim is None:
            self.self_attn = True
            self.key = nn.Linear(hidden_dim, embed_dim, bias=False)
            self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        else:
            self.self_attn = False
            self.key = nn.Linear(context_dim, embed_dim, bias=False)
            self.value = nn.Linear(context_dim, hidden_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(embed_dim, hidden_dim, bias=True))

    def forward(self, tokens, context=None):
        """_summary_

        Args:
            tokens (_type_): [batch, sequence_len, hidden_dim]
            context (_type_, optional): [batch, contex_seq_len, context_dim]. Defaults to None.
        """
        Q = self.query(tokens)
        K = self.key(tokens) if self.self_attn else self.key(context)
        V = self.value(tokens) if self.self_attn else self.value(context)

        if self.debug:
            print("Q:{}, K:{},V:{}".format(Q.shape, K.shape, V.shape))

        # transform heads onto batch dimension
        Q = rearrange(Q, "B T (H D) -> (B H) T D", H=self.num_heads, D=self.head_dim)
        K = rearrange(K, "B T (H D) -> (B H) T D", H=self.num_heads, D=self.head_dim)
        V = rearrange(V, "B T (H D) -> (B H) T D", H=self.num_heads, D=self.head_dim)
        if self.debug:
            print("Q:{}, K:{},V:{}".format(Q.shape, K.shape, V.shape))
        scoremats = torch.einsum("BTD,BSD->BTS", Q, K)
        attnmats = F.softmax(scoremats / math.sqrt(self.head_dim), dim=-1)
        # print(scoremats.shape, attnmats.shape, )
        ctx_vecs = torch.einsum("BTS,BSD->BTD", attnmats, V)
        # split the heads transform back to hidden.
        ctx_vecs = rearrange(
            ctx_vecs, "(B H) T D -> B T (H D)", H=self.num_heads, D=self.head_dim
        )

        return self.to_out(ctx_vecs)
