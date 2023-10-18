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
        """Naming of all layers are same as in huggingface model
            becouse we want to initialize our model with pretrained model

        Args:
            num_in_channels (int): the number of input channels
            emb_time_dim (int): the size of timestep embeddings
            out_channels (_type_, optional): is the number of out channels. defaults to `num_in_channels.
        """
        super().__init__(*args, **kwargs)

        if out_channels is None:
            out_channels = num_in_channels

        self.in_channel = num_in_channels
        self.time_emb_dim = emb_time_dim
        self.out_channel = out_channels

        self.norm1 = nn.GroupNorm(32, self.in_channel, eps=1e-05, affine=True)
        self.conv1 = nn.Conv2d(
            self.in_channel, self.out_channel, kernel_size=3, stride=1, padding=1
        )
        self.time_emb_proj = nn.Linear(
            in_features=self.time_emb_dim, out_features=self.out_channel, bias=True
        )
        self.norm2 = nn.GroupNorm(32, self.out_channel, eps=1e-05, affine=True)
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.conv2 = nn.Conv2d(
            self.out_channel, self.out_channel, kernel_size=3, stride=1, padding=1
        )
        self.nonlinearity = nn.SiLU()
        if self.in_channel == self.out_channel:
            self.conv_shortcut = nn.Identity()
        else:
            self.conv_shortcut = nn.Conv2d(
                self.in_channel, self.out_channel, kernel_size=1, stride=1
            )

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor, cross_attention_kwargs=None
    ):
        """_summary_

        Args:
            x (torch.Tensor): is the input feature map with shape [batch_size, channels, height, width]
            t_emb (torch.Tensor): is the time step embeddings of shape [batch_size, d_t_emb]
            cond (None): _description_
        """
        # Input conv
        h = self.norm1(x)
        h = self.nonlinearity(h)
        h = self.conv1(h)

        # Add time step emb
        if t_emb is not None:
            t_hidden = self.time_emb_proj(self.nonlinearity(t_emb))
            h = h + t_hidden[:, :, None, None]

        # Output conv
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Skip connection
        return h + self.conv_shortcut(x)
