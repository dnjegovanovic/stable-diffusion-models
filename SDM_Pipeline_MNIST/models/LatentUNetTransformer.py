import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .GaussianFourierProjection import GFP
from .ReshapeOutputToFM import RODense
from .SpatialTransformer import SpatialTransformer


class LatentUnetTransformerModel(nn.Module):
    def __init__(
        self,
        marginal_prob_std,
        channels=[4, 64, 128, 256],
        embed_dim=256,
        text_dim=256,
        n_class=10,
        *args,
        **kwargs
    ) -> None:
        """_summary_

        Args:
            marginal_prob_std (_type_): A function that takes time t and gives the standard
                                    deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
            channels (list, optional): The number of channels for feature maps of each resolution.. Defaults to [32, 64, 128, 256].
            embed_dim (int, optional): The dimensionality of Gaussian random feature embeddings of time. Defaults to 256.
            text_dim (int, optional): the embedding dimension of text / digits. Defaults to 256.
            n_class (int, optional): number of classes you want to model. Defaults to 10.
        """
        super().__init__(*args, **kwargs)

        # Gaussian random feature embedding layer for time
        self.time_embeded = nn.Sequential(
            GFP(embeded_dim=embed_dim), nn.Linear(embed_dim, embed_dim)
        )

        # Encoding layers where the resolution decreases(can be implelemted as separated block)
        self.conv_1 = nn.Conv2d(channels[0], channels[1], 3, stride=1, bias=False)
        self.dense_1 = RODense(embed_dim, channels[1])
        self.gnorm_1 = nn.GroupNorm(4, num_channels=channels[1])

        self.conv_2 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense_2 = RODense(embed_dim, channels[2])
        self.gnorm_2 = nn.GroupNorm(4, num_channels=channels[2])
        self.attn_2 = SpatialTransformer(channels[2], text_dim)

        self.conv_3 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense_3 = RODense(embed_dim, channels[3])
        self.gnorm_3 = nn.GroupNorm(4, num_channels=channels[3])
        self.attn_3 = SpatialTransformer(channels[3], text_dim)

        # self.conv_4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        # self.dense_4 = RODense(embed_dim, channels[3])
        # self.gnorm_4 = nn.GroupNorm(4, num_channels=channels[3])
        # self.attn_4 = SpatialTransformer(channels[3], text_dim)

        # Decoding layers where the resolution increases(can be implelemted as separated block)
        # self.tconv_4 = nn.ConvTranspose2d(
        #     channels[3], channels[2], 3, stride=2, bias=False
        # )
        # self.dense_5 = RODense(embed_dim, channels[2])
        # self.tgnorm_4 = nn.GroupNorm(32, num_channels=channels[2])

        self.tconv_3 = nn.ConvTranspose2d(
            channels[3],
            channels[2],
            3,
            stride=2,
            bias=False,
        )  #  + channels[2]
        self.dense_6 = RODense(embed_dim, channels[2])
        self.tgnorm_3 = nn.GroupNorm(4, num_channels=channels[2])
        self.attn_6 = SpatialTransformer(channels[2], text_dim)

        self.tconv_2 = nn.ConvTranspose2d(
            channels[2], channels[1], 3, stride=2, bias=False, output_padding=1
        )  #  + channels[1]
        self.dense_7 = RODense(embed_dim, channels[1])
        self.tgnorm_2 = nn.GroupNorm(4, num_channels=channels[1])
        self.tconv_1 = nn.ConvTranspose2d(
            channels[1], channels[0], 3, stride=1
        )  #  + channels[0]

        # The swish activation function
        self.act = nn.SiLU()  # lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std
        self.cond_embed = nn.Embedding(n_class, text_dim)

    def forward(self, x: torch.Tensor, time_feature: torch.Tensor, y=None):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.time_embeded(time_feature))
        y_embed = self.cond_embed(y).unsqueeze(1)
        # Encoding path
        h1 = self.conv_1(x) + self.dense_1(embed)
        ## Incorporate information from t
        ## Group normalization
        h1 = self.act(self.gnorm_1(h1))
        h2 = self.conv_2(h1) + self.dense_2(embed)
        h2 = self.act(self.gnorm_2(h2))

        h2 = self.attn_2(h2, y_embed)
        h3 = self.conv_3(h2) + self.dense_3(embed)
        h3 = self.act(self.gnorm_3(h3))
        h3 = self.attn_3(h3, y_embed)

        # h4 = self.conv_4(h3) + self.dense_4(embed)
        # h4 = self.act(self.gnorm_4(h4))
        # h4 = self.attn_4(h4, y_embed)

        # Decoding path
        h = self.tconv_3(h3) + self.dense_6(embed)
        ## Skip connection from the encoding path
        h = self.act(self.tgnorm_3(h))
        h = self.attn_6(h, y_embed)
        h = self.tconv_2(h + h2)
        h += self.dense_7(embed)
        h = self.act(self.tgnorm_2(h))
        h = self.tconv_1(h + h1)

        # Normalize output
        h = h / self.marginal_prob_std(time_feature)[:, None, None, None]
        return h
