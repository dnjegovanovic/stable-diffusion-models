import torch
import torch.nn as nn

from .GaussianFourierProjection import *
from .ReshapeOutputToFM import *


class SimpleUNet(nn.Module):
    def __init__(
        self,
        marginal_prob_std,
        channels=[32, 64, 128, 256],
        embed_dim: int = 256,
        *args,
        **kwargs
    ) -> None:
        """Time-dependent score-based network.

        Args:
            marginal_prob_std (_type_): A function that takes time t and gives the standard
                                        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
            channels (list, optional): The number of channels for feature maps of each resolution.. Defaults to [32, 64, 128, 256].
            embed_dim (int, optional): The dimensionality of Gaussian random feature embeddings. Defaults to 256.
        """
        super().__init__(*args, **kwargs)
        self.chanels = channels
        self.embed_dim = embed_dim
        self.marginal_prob_std = marginal_prob_std

        self.time_embed = nn.Sequential(
            GFP(embeded_dim=self.embed_dim), nn.Linear(self.embed_dim, self.embed_dim)
        )

        # Encoding layers, downsampling resolution
        self.conv_1 = nn.Conv2d(1, self.chanels[0], kernel_size=3, stride=1, bias=False)
        self.ro_dense_1 = RODense(self.embed_dim, self.chanels[0])
        self.gnorm_1 = nn.GroupNorm(4, num_channels=self.chanels[0])

        self.conv_2 = nn.Conv2d(
            self.chanels[0], self.chanels[1], kernel_size=3, stride=2, bias=False
        )
        self.ro_dense_2 = RODense(self.embed_dim, self.chanels[1])
        self.gnorm_2 = nn.GroupNorm(32, num_channels=self.chanels[1])

        self.conv_3 = nn.Conv2d(
            self.chanels[1], self.chanels[2], kernel_size=3, stride=2, bias=False
        )
        self.ro_dense_3 = RODense(self.embed_dim, self.chanels[2])
        self.gnorm_3 = nn.GroupNorm(32, num_channels=self.chanels[2])

        self.conv_4 = nn.Conv2d(
            self.chanels[2], self.chanels[3], kernel_size=3, stride=2, bias=False
        )
        self.ro_dense_4 = RODense(self.embed_dim, self.chanels[3])
        self.gnorm_4 = nn.GroupNorm(32, num_channels=self.chanels[3])
        ######################################

        # Decoding layers, increases resolution
        self.trans_conv_4 = nn.ConvTranspose2d(
            self.chanels[3], self.chanels[2], kernel_size=3, stride=2, bias=False
        )
        self.ro_dense_5 = RODense(self.embed_dim, self.chanels[2])
        self.trans_gnorm_4 = nn.GroupNorm(32, num_channels=self.chanels[2])

        self.trans_conv_3 = nn.ConvTranspose2d(
            self.chanels[2] + self.chanels[2],
            self.chanels[1],
            kernel_size=3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.ro_dense_6 = RODense(self.embed_dim, self.chanels[1])
        self.trans_gnorm_3 = nn.GroupNorm(32, num_channels=self.chanels[1])

        self.trans_conv_2 = nn.ConvTranspose2d(
            self.chanels[1] + self.chanels[1],
            self.chanels[0],
            kernel_size=3,
            stride=2,
            bias=False,
            output_padding=1,
        )
        self.ro_dense_7 = RODense(self.embed_dim, self.chanels[0])
        self.trans_gnorm_2 = nn.GroupNorm(32, num_channels=self.chanels[0])

        self.trans_conv_1 = nn.ConvTranspose2d(
            self.chanels[0] + self.chanels[0], 1, kernel_size=3, stride=1
        )

        self.act_fun = lambda x: x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor, time_feature: torch.Tensor, y=None):
        """_summary_

        Args:
            x (torch.Tensor): input data
            time_feature (torch.Tensor):time data obtain the gaussina radnom feature embeding for t
            y (_type_, optional): _description_. Defaults to None.
        """

        # Gaussian random time feature embedding
        embed = self.act_fun(self.time_embed(time_feature))

        # Encoding part and incorporate time infromation
        # Downsampling
        h1 = self.conv_1(x) + self.ro_dense_1(embed)
        h1 = self.act_fun(self.gnorm_1(h1))

        h2 = self.conv_2(h1) + self.ro_dense_2(embed)
        h2 = self.act_fun(self.gnorm_2(h2))

        h3 = self.conv_3(h2) + self.ro_dense_3(embed)
        h3 = self.act_fun(self.gnorm_3(h3))

        h4 = self.conv_4(h3) + self.ro_dense_4(embed)
        h4 = self.act_fun(self.gnorm_4(h4))

        # Encoding feature, SubSampling
        h = self.trans_conv_4(h4)
        # Add skip connection
        h += self.ro_dense_5(embed)
        h = self.act_fun(self.trans_gnorm_4(h))
        h = self.trans_conv_3(torch.cat([h, h3], dim=1))

        h += self.ro_dense_6(embed)
        h = self.act_fun(self.trans_gnorm_3(h))
        h = self.trans_conv_2(torch.cat([h, h2], dim=1))

        h += self.ro_dense_7(embed)
        h = self.act_fun(self.trans_gnorm_2(h))
        h = self.trans_conv_1(torch.cat([h, h1], dim=1))

        # Normalize output
        h = h / self.marginal_prob_std(time_feature)[:, None, None, None]

        return h
