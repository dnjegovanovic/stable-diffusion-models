import torch
import torch.nn as nn
import torch.nn.functional as F


class UpSample(nn.Module):
    """Up-sampling layer."""

    def __init__(
        self, num_chanles: int, scale_factor: int, mode="nearest", *args, **kwargs
    ) -> None:
        """_summary_

        Args:
            num_chanles (int): number of chanels
            scale_factor (int): scale factor for interpolation fun
            mode (str, optional): interpolation mode 'nearest'.
        """
        super().__init__(*args, **kwargs)

        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(
            num_chanles, num_chanles, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return self.conv(x)
