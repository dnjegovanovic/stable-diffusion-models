import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class UNetModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)