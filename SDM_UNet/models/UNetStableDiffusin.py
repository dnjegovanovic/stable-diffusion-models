import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from collections import OrderedDict
from easydict import EasyDict as edict
from typing import Tuple, List

from .TimeModulation import *
from .SpatialTransformer import *
from .DownlSampling import *
from .UpSampling import *


class UNetStableDiffusion(nn.Module):
    def __init__(
        self,
        num_in_channels: int = 4,
        channels: int = 320,
        n_res_block: int = 2,
        time_emb_dim: int = 1280,
        context_dim: int = 768,
        channel_multipliers: Tuple(int) = (1, 2, 4, 4),
        attention_levels: Tuple(int) = (0, 1, 2),
        num_attention_heads: int = 8,
        *args,
        **kwargs
    ) -> None:
        """Assamble all component for UNet arch

        Args:
            num_in_channels (int, optional): is the number of channels in the input feature map. Defaults to 4.
            channels (int, optional): is the base channel count for the model. Defaults to 320.
            n_res_block (int, optional): number of residual blocks at each level. Defaults to 2.
            time_emb_dim (int, optional): Size time embeddings. Defaults to 1280.
            context_dim (int, optional): cross attention context dimension. Defaults to 768.
            channel_multipliers (Tuple, optional): are the multiplicative factors for number of channels for each level. Defaults to (1,2,4,4).
            attention_levels (Tuple, optional): are the levels at which attention should be performed. Defaults to (0,1,2).
            num_attention_heads (int, optional): is the number of attention heads in the transformers. Defaults to 8.
        """
        super().__init__(*args, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_in_channels = num_in_channels
        self.num_out_chanels = num_in_channels
        self.channels = channels
        self.n_res_block = n_res_block
        self.time_emb_dim = time_emb_dim
        self.context_dim = context_dim
        self.channel_multipliers = channel_multipliers
        self.attention_levels = attention_levels
        self.num_attention_heads = num_attention_heads

        # Number of levels
        n_level = len(channel_multipliers)

        # Number of channels at each level
        level_chanel_list = [channels * mult for mult in channel_multipliers]

        # Size time embeddings
        # create embedding layer
        self.time_embedding = nn.Sequential(
            OrderedDict(
                {
                    "linear_1": nn.Linear(channels, time_emb_dim, bias=True),
                    "act_fun": nn.SiLU(),
                    "linear_2": nn.Linear(time_emb_dim, time_emb_dim, bias=True),
                }
            )
        )

        # Initial 3×3 convolution that maps the input to channels
        self.conv_input = nn.Conv2d(
            self.num_in_channels, self.channels, kernel_size=3, stride=1, padding=1
        )

        # Downsampling blocks
        self.down_blocks = TimeModulatedSeqModel()
        self.down_blocks_chanels = [channels]  # 320
        curr_chan_stats = channels

        for i in range(n_level):  # Prepare levels
            for j in range(self.n_res_block):  # Add the residual blocks and attentions
                res_attn = TimeModulatedSeqModel()
                # Residual block maps from previous number of channels
                # to the number of channels in the current level
                # input_chan of first ResBlock is different from the rest

                res_attn.append(
                    ResBlock(
                        num_in_channels=curr_chan_stats,
                        emb_time_dim=time_emb_dim,
                        out_channels=level_chanel_list[i],
                    )
                )

                # add attention on specfyed level, Add transformer
                if i in self.attention_levels:
                    res_attn.append(
                        SpatialTransformer(
                            hidden_dim=level_chanel_list[i],
                            context_dim=self.context_dim,
                            num_heads=self.num_attention_heads,
                        )
                    )

                # Add them to the input half of the U-Net and
                # keep track of the number of channels of its output
                curr_chan_stats = level_chanel_list[i]
                self.down_blocks.append(res_attn)
                self.down_blocks_chanels.append(curr_chan_stats)

                # Down sample at all levels except last
                if not i == n_level - 1:
                    self.down_blocks.append(
                        TimeModulatedSeqModel(
                            DownSaple(num_chanels=level_chanel_list[i])
                        )
                    )
                    self.down_blocks_chanels.append(curr_chan_stats)

        # The middle of the U-Net
        self.middle_block = TimeModulatedSeqModel(
            ResBlock(num_in_channels=curr_chan_stats, emb_time_dim=time_emb_dim),
            SpatialTransformer(
                hidden_dim=curr_chan_stats, context_dim=self.context_dim
            ),
            ResBlock(num_in_channels=curr_chan_stats, emb_time_dim=time_emb_dim),
        )

        # Second half of the U-Net
        # Tensor Upsample blocks
        self.upsample_output_blocks = nn.ModuleList()

        for i in reversed(range(n_level)):  # Prep levels in reversed order
            for j in range(self.n_res_block + 1):  # Add residual blocks and attention
                res_attn = TimeModulatedSeqModel()

                # Residual block maps from previous number of channels plus
                # the skip connections from the input half of U-Net
                # to the number of channels in the current level.
                res_attn.append(
                    ResBlock(
                        num_in_channels=curr_chan_stats
                        + self.down_blocks_chanels.pop(),
                        emb_time_dim=time_emb_dim,
                        out_channels=level_chanel_list[i],
                    )
                )

                # Add transformer
                if i in self.attention_levels:
                    res_attn.append(
                        SpatialTransformer(
                            hidden_dim=level_chanel_list[i],
                            context_dim=self.context_dim,
                        )
                    )

                # Up-sample at every level after last residual block except the last one.
                # Note that we are iterating in reverse; i.e. i == 0 is the last.

                if j == n_res_block and i != 0:
                    res_attn.append(UpSample(num_chanles=level_chanel_list[i]))

                self.upsample_output_blocks.append(res_attn)

        # Final normalization and 3×3 convolution

        self.output = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.num_out_chanels, kernel_size=3, padding=1),
        )

        self.to(self.device)

    def time_step_proj_embedding(
        self, time_step: torch.Tensor, max_period: int = 10000
    ):
        """Create sinusoidal time step embeddings

        Args:
            time_step (torch.Tensor): are the time steps of shape [batch_size]
            max_period (int, optional): controls the minimum frequency of the embeddings.. Defaults to 10000.
        """

        half = self.channels // 2  # half the channels are sin and the other half is cos
        if time_step.ndim == 0:
            time_step = time_step.unsqueeze(0)

        # a = t/exp(10000,(2i/chnels))
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=time_step.device)

        angles = time_step[:, None].float() * frequencies[None, :]
        # cos(a) nad sin(a)
        return torch.cat(
            [torch.cos(angles), torch.sin(angles)], dim=-1
        )  # half cos and half sin

    def forward(
        self,
        x: torch.Tensor,
        time_step: torch.Tensor,
        cond=None,
        encoder_hidden_states=None,
        output_dict=True,
    ):
        """_summary_

        Args:
            x (torch.Tensor): is the input feature map of shape [batch_size, channels, width, height]
            time_step (torch.Tensor): are the time steps of shape [batch_size]
            cond (_type_, optional): conditioning of shape [batch_size, n_cond, d_cond]. Defaults to None.
            encoder_hidden_states (_type_, optional): _description_. Defaults to None.
            output_dict (bool, optional): To store the input half outputs for skip connections. Defaults to True.
        """

        if cond is None and encoder_hidden_states is not None:
            cond = encoder_hidden_states

        # Get time step embeddings
        t_emb = self.time_step_proj_embedding(time_step=time_step)
        t_emb = self.time_embedding(t_emb)

        # Input half of the U-Net
        x = self.conv_input(x)
        down_x_cache = [x]
        for module in self.down_blocks:
            x = module(x, t_emb, cond)
            down_x_cache.append(x)

        # Middle of the UNet
        x = self.middle_block(x, t_emb, cond)

        # Output half od the UNet
        for module in self.upsample_output_blocks:
            x = module(torch.cat((x, down_x_cache.pop()), dim=1), t_emb, cond)

        # Final normalization and 3×3 convolution
        x = self.output(x)
        if output_dict:
            return edict(sample=x)
        else:
            return x
