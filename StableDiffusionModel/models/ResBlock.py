import torch
import torch.nn as nn

"""_summary_
Implement ResBlock as a part of backbone U-Net
"""


class ResBlock(nn.Module):
    def __init__(
        self,
        num_in_channels: int,
        emb_time_dim: int,
        out_channels=None,
        *args,
        **kwargs
    ) -> None:
        """_summary_

        Args:
            num_in_channels (int): the number of input channels
            emb_time_dim (int): the size of timestep embeddings
            out_channels (_type_, optional): is the number of out channels. defaults to `num_in_channels.
        """
        super().__init__(*args, **kwargs)

        if out_channels is None:
            out_channels = num_in_channels

        self.num_in_channels = num_in_channels
        self.emb_time_dim = emb_time_dim
        self.out_channels = out_channels

        self.norm_1 = nn.GroupNorm(32, num_in_channels, eps=1e-05, affine=True)
        self.conv_1 = nn.Conv2d(
            num_in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.emb_time_proj = nn.Linear(
            in_features=emb_time_dim, out_features=out_channels, bias=True
        )
        self.norm_2 = nn.GroupNorm(32, out_channels, eps=1e-05, affine=True)
