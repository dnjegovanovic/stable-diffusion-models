import torch

import unittest

import matplotlib.pyplot as plt

from SDM_Pipeline_MNIST.models.AutoEncoder import *


class TestForwardReverseDiffusionFun(unittest.TestCase):
    def test_autoencoder(self):
        x_tmp = torch.randn(1, 1, 28, 28)
        print(AutoEncoder()(x_tmp).shape)
        assert (
            AutoEncoder()(x_tmp).shape == x_tmp.shape
        ), "Check conv layer spec! the autoencoder input output shape not align"
