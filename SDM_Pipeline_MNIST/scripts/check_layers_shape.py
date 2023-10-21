import torch
import torch.nn as nn


def check_layer_shape():
    net = nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=1, bias=False),
        nn.ConvTranspose2d(32, 1, 3, stride=1),
    )
    x = torch.randn((1, 1, 28, 28))
    for layer in net:
        x = layer(x)
        print(x.shape)


if __name__ == "__main__":
    check_layer_shape()
