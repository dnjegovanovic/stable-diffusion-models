import torch
import torch.nn as nn

import numpy as np

class GFP(nn.Module):
    """The code below helps our neural network synchronize with time by timing. 
    The idea is that instead of just telling our network a quantity (the current time), 
    we will express the current time in terms of a number of sinusoidal objects hopefully, 
    if we tell our network times the current time more likely to respond to change over time.
    This allows us to efficiently explore the time dependence of the score function s(x, t).
    """
    def __init__(self,embeded_dim, scale=30, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Randomly sample weights (frequencies) during initialization. 
        # These weights (frequencies) are fixed during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embeded_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        # Cosine(2 pi freq x), Sine(2 pi freq x)
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)