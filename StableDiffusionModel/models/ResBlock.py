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

        # Normalize convolution
        self.norm_1 = nn.GroupNorm(32, num_in_channels, eps=1e-05, affine=True)
        self.conv_1 = nn.Conv2d(
            num_in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        # Time step embeddings
        self.emb_time_proj = nn.Linear(
            in_features=emb_time_dim, out_features=out_channels, bias=True
        )
        # Final conv layer
        self.norm_2 = nn.GroupNorm(32, out_channels, eps=1e-05, affine=True)
        self.dropout = nn.Dropout2d(p=0.0, inplace=False)
        self.conv_2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.nonlinearity = nn.SiLU()

        # chanel to out_chanel mapping layer for residual
        if num_in_channels == out_channels:
            self.conv_shortcut = nn.Identity()
        else:
            self.conv_shortcut = nn.Conv2d(
                num_in_channels, out_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        """_summary_

        Args:
            x (torch.Tensor): is the input feature map with shape [batch_size, channels, height, width]
            t_emb (torch.Tensor): is the time step embeddings of shape [batch_size, d_t_emb]
            cond (None): _description_
        """
        # Input conv
        h = self.norm_1(x)
        h = self.nonlinearity(h)
        h = self.conv_1(h)

        # Add time step emb
        if t_emb is not None:
            t_hidden = self.emb_time_proj(self.nonlinearity(t_emb))
            h = h + t_hidden[:, :, None, None]

        # Output conv
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Skip connection
        return h + self.conv_shortcut(x)
