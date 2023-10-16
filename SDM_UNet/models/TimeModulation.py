import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .ResBlock import *
from .SpatialTransformer import *


class TimeModulatedSeqModel(nn.Sequential):
    # Analize models and involve time modulation in some layer
    def forward(self, x, time_emb, cond=None, cross_attention_kwargs=None):
        for module in self:
            if isinstance(module, TimeModulatedSeqModel):
                x = module(x, time_emb, cond)

            elif isinstance(module, ResBlock):
                # For some layer add modulation
                x = module(x, time_emb)
            elif isinstance(module, SpatialTransformer):
                # for some layer add class cond
                x = module(x, cond)

            else:
                x = module(x)

        return x
