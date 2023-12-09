import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.datasets import MNIST
import torchvision.transforms as transfroms
from torch.utils.data import DataLoader
import torch.utils.data as data
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

import numpy as np
import functools

from SDM_Pipeline_MNIST.models.UnetTransformerModel import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UNetTransformer(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.__dict__.update(kwargs)
        self.batch_size = self.UnetTR["batch_size"]
        self.num_epochs = self.UnetTR["num_epochs"]
        self.sigma = self.UnetTR["sigma"]
        self.euler_maruyam_num_steps = self.UnetTR["euler_maruyam_num_steps"]
        self.eps_stab = self.UnetTR["eps_stab"]
        self.lr = self.UnetTR["lr"]

        self._setup_arch()
        self._setup_data()
        self.save_hyperparameters()

    def _setup_arch(self):
        self.marg_prob_fun = functools.partial(
            self._marginal_prob_std, sigma=self.sigma
        )

        self.model = UnetTransformerModel(self.marg_prob_fun)
        self.model.to(device)
        self.model_params = list(self.model.parameters())

    def _setup_data(self):
        dataset = MNIST(".", train=True, transform=transfroms.ToTensor(), download=True)
        # use 20% of training data for validation
        train_set_size = int(len(dataset) * 0.95)
        valid_set_size = len(dataset) - train_set_size

        # split the train set into two
        seed = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = data.random_split(
            dataset, [train_set_size, valid_set_size], generator=seed
        )

    def _marginal_prob_std(self, time_step, sigma) -> torch.Tensor:
        """Compute the mean and standard deviation of p_{0t}(x(t) | x(0)).
            A function that gives the standard deviation of the perturbation kernel

        Args:
            time_step (_type_): vector of time step
            sigma (_type_): sigma in SDE

        Returns:
            torch.Tensor: standard deviation
        """

        t = torch.tensor(time_step, device=device)
        std = torch.sqrt((sigma ** (2 * t) - 1.0) / 2.0 / np.log(sigma))

        return std

    def _diffusion_coeff(self, time_step, sigma) -> torch.Tensor:
        """Compute the diffusion coefficient of our SDE.

        Args:
            time_step (_type_): Time step ector
            sigma (_type_): The sigma in our SDE.

        Returns:
            torch.Tensor: The vector of diffusion coefficients.
        """

        return torch.tensor(sigma**time_step, device=device)

    def _loss_fn(self, sample: torch.Tensor, eps=1e-5):
        """The loss function for training score-based generative models.

        Args:
            sample (_type_): A mini-batch of training data
            eps (_type_, optional): A tolerance value for numerical stability. Defaults to 1e-5.
        """

        # Sample time uniformly from 0 to 1
        random_t = (
            torch.rand(sample[0].shape[0], device=sample[0].device) * (1.0 - eps) + eps
        )
        # Fine the noise std at the time t
        std = self._marginal_prob_std(random_t, self.sigma)
        z = torch.randn_like(sample[0])  # get normally distributed noise
        perturbed_x = sample[0] + std[:, None, None, None] * z

        score = self.model(perturbed_x, random_t, sample[1])
        loss = torch.mean(
            torch.sum((score * std[:, None, None, None] + z) ** 2, dim=(1, 2, 3))
        )
        return loss

    def _euler_maruyam_sampler(self, eps=1e-3, y=None, x_shape=(1, 28, 28)):
        """The Euler–Maruyama method (also called the Euler method) is a method for the approximate numerical
        solution of a stochastic differential equation (SDE). It is an extension of the Euler method for ordinary
        differential equations to stochastic differential equations. It is named after Leonhard Euler and Gisiro Maruyama.
        Unfortunately, the same generalization cannot be done for any arbitrary deterministic method.

        Generate samples from score-based models with the Euler-Maruyama solver

        Args:
            eps (_type_, optional): The smallest time step for numerical stability.. Defaults to 1e-3.
            y (_type_, optional): _description_. Defaults to None.
            euler_maruyam_num_steps: The number of sampling steps. Equivalent to the number of discretized time steps.
        """

        t = torch.ones(self.batch_size, device=device)
        init_x = (
            torch.randn(self.batch_size, *x_shape, device=device)
            * self._marginal_prob_std(t, self.sigma)[:, None, None, None]
        )
        time_steps = torch.linspace(
            1.0, eps, self.euler_maruyam_num_steps, device=device
        )
        step_size = time_steps[0] - time_steps[1]
        x = init_x
        with torch.no_grad():
            for time_step in time_steps:
                batch_time_step = torch.ones(self.batch_size, device=device) * time_step
                g = self._diffusion_coeff(batch_time_step, self.sigma)
                mean_x = (
                    x
                    + (g**2)[:, None, None, None]
                    * self.model(x, batch_time_step, y=y)
                    * step_size
                )
                x = mean_x + torch.sqrt(step_size) * g[
                    :, None, None, None
                ] * torch.randn_like(x)

        return mean_x

    def forward(self, x):
        # Sample time uniformly in 0, 1
        random_t = (
            torch.rand(x[0].shape[0], device=x.device) * (1.0 - self.eps_stab)
            + self.eps_stab
        )
        # Find the noise std at the time `t`
        std = self._marginal_prob_std(random_t, sigma=self.sigma)
        z = torch.randn_like(x[0])  # get normally distributed noise
        perturbed_x = x[0] + std[:, None, None, None] * z
        score = self.model(perturbed_x, random_t, x[1])

        return score

    def training_step(self, sample, batch_idx):
        loss = self._loss_fn(sample)
        self.log("train_loss", loss.item(), prog_bar=True)
        return loss

    def validation_step(self, sample, batch_idx):
        val_loss = self._loss_fn(sample)
        self.log("val_loss", val_loss.item(), prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        opt_model = Adam([{"params": self.model_params}], lr=self.lr)
        scheduler = LambdaLR(
            opt_model, lr_lambda=lambda epoch: max(0.2, 0.98**self.num_epochs)
        )
        return [opt_model], {"scheduler": scheduler}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            timeout=30,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            timeout=30,
        )
