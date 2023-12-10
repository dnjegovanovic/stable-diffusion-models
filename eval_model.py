import torch
import torch.nn as nn
import torch.utils.data as data

from pathlib import Path

from argparse import ArgumentParser
from SDM_Pipeline_MNIST.modules.SimpleUnetModules import *
from SDM_Pipeline_MNIST.modules.UnetTransformerModules import *
from SDM_Pipeline_MNIST.misc.visualize_internal_layer import visualize_digit_embedding
from SDM_Pipeline_MNIST.modules.AutoEncoderModules import *

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


def test_autoencoder(args):
    save_path = Path(
        "D:/ML_AI_DL_Projects/projects_repo/stable-diffusion-models/playground_imgs/SDM_Pipeline_MNIST_imgs"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = AutoEncoderModule.load_from_checkpoint(args.model)
    module.to(device)
    module.eval()
    data_loader = DataLoader(module.val_dataset, batch_size=module.batch_size, shuffle=False, num_workers=4)

    x, y = next(iter(data_loader))
    x_hat = module(x.to(device)).cpu()
    
    plt.figure(figsize=(6,6.5))
    plt.axis('off')
    plt.imshow(make_grid(x[:64,:,:,:].cpu()).permute([1,2,0]), vmin=0., vmax=1.)
    plt.title("Original")
    plt.savefig(save_path / "ae_original.png")

    plt.figure(figsize=(6,6.5))
    plt.axis('off')
    plt.imshow(make_grid(x_hat[:64,:,:,:].cpu()).permute([1,2,0]), vmin=0., vmax=1.)
    plt.title("AE Reconstructed")
    plt.savefig(save_path / "ae_recnstructed.png")
    
    

@torch.no_grad()
def main(args):
    if args.test_simpleunet:
        test_simple_Unet(args)
    elif args.test_unettransforme:
        test_UnetTransformer(args)
    elif args.test_autoencoder:
        test_autoencoder(args)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--model", required=True, help="model to evaluate")
    parser.add_argument(
        "--test-unettransforme", action="store_true", help="Transformer Model"
    )
    parser.add_argument("--test-simpleunet", action="store_true", help="UNet Model")
    parser.add_argument("--test-autoencoder", action="store_true", help="AutoEncoder Model") 
    main(parser.parse_args())
