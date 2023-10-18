import torch
import torch.nn as nn
import torch.nn.functional as F


class DownSample(nn.Module):
    def __init__(self, num_chanels: int, *args, **kwargs) -> None:
        """Down sampling feature

        Args:
            num_chanels (int): number of chanesl
        """
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(
            num_chanels, num_chanels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor, cross_attention_kwargs=None):
        return self.conv(x)
