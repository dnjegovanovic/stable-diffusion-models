import torch
import torch.nn as nn
import torch.utils.data as data

from pathlib import Path

from argparse import ArgumentParser
from SDM_Pipeline_MNIST.modules.SimpleUnetModules import *
from SDM_Pipeline_MNIST.modules.UnetTransformerModules import *
from SDM_Pipeline_MNIST.misc.visualize_internal_layer import visualize_digit_embedding

import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def test_simple_Unet(args):
    save_path = Path(
        "D:/ML_AI_DL_Projects/projects_repo/stable-diffusion-models/playground_imgs/SDM_Pipeline_MNIST_imgs"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = SimpleUNetModules.load_from_checkpoint(args.model)
    module.to(device)
    module.eval()
    sample_num = 64
    module.batch_size = sample_num
    samples = module._euler_maruyam_sampler()

    samples = samples.clamp(0.0, 1.0)
    sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_num)))

    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0.0, vmax=1.0)
    plt.savefig(save_path / "samples_scorebased_Unet.png")


def test_UnetTransformer(args):
    save_path = Path(
        "D:/ML_AI_DL_Projects/projects_repo/stable-diffusion-models/playground_imgs/SDM_Pipeline_MNIST_imgs"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = UNetTransformer.load_from_checkpoint(args.model)
    module.to(device)
    module.eval()
    sample_num = 64
    digit = 4
    module.batch_size = sample_num
    y = digit * torch.ones(sample_num, dtype=torch.long)
    samples = module._euler_maruyam_sampler(y=y.to(device))

    samples = samples.clamp(0.0, 1.0)
    sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_num)))

    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0.0, vmax=1.0)
    plt.savefig(save_path / "samples_scorebased_Unet.png")

    visualize_digit_embedding(module.model.cond_embed.weight.data, save_path)


@torch.no_grad()
def main(args):
    if args.test_simpleunet:
        test_simple_Unet(args)
    elif args.test_unettransforme:
        test_UnetTransformer(args)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--model", required=True, help="model to evaluate")
    parser.add_argument(
        "--test-unettransforme", action="store_true", help="Transformer Model"
    )
    parser.add_argument("--test-simpleunet", action="store_true", help="UNet Model")
    main(parser.parse_args())
