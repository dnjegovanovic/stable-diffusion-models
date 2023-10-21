import torch
import torch.nn as nn

import numpy as np


class RODense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps.
    Allow time representation to input additively from the side of a convolution layer
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]
        # this broadcast the 2d tensor to 4d, add the same value across space.
