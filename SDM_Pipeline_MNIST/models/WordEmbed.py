import torch
import torch.nn as nn


class WordEmbed(nn.Module):
    def __init__(self, vocab_size, embed_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embed_word = nn.Embedding(vocab_size + 1, embed_dim)

    def forward(self, x):
        return self.embed_word(x)
