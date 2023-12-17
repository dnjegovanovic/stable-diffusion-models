import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data
from torch.optim import Adam
from torch.utils.data import TensorDataset

import pytorch_lightning as pl


from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from lpips import LPIPS

from SDM_Pipeline_MNIST.models.AutoEncoder import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AutoEncoderModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.__dict__.update(kwargs)

        self.batch_size = self.AE["batch_size"]
        self.num_epochs = self.AE["num_epochs"]
        self.lr = self.AE["lr"]

        self.lp_loss = LPIPS(net="squeeze").to(device)

        self._setup_arch()
        self._setup_data()
        self.save_hyperparameters()

    def _setup_arch(self):
        self.model = AutoEncoder([4, 4, 4])
        self.model.to(device)
        self.model_params = list(self.model.parameters())

    def _setup_data(self):
        dataset = MNIST(".", train=True, transform=transforms.ToTensor(), download=True)
        # use 20% of training data for validation
        train_set_size = int(len(dataset) * 0.95)
        valid_set_size = len(dataset) - train_set_size

        # split the train set into two
        seed = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = data.random_split(
            dataset, [train_set_size, valid_set_size], generator=seed
        )

    def _loss_fn(self, x: torch.Tensor, x_hat: torch.Tensor):
        mse = nn.functional.mse_loss(x, x_hat)
        lp = self.lp_loss(x.repeat(1, 3, 1, 1), x_hat.repeat(1, 3, 1, 1)).mean()

        return mse + lp

    def create_latent_sapce(self, batch_size, save_data=False):
        full_dataset = data.ConcatDataset([self.train_dataset, self. val_dataset])
        data_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.model.requires_grad_(False)
        self.model.eval()
        
        zs = []
        ys = []
        for x, y in data_loader:
            z = self.model.encoder(x.to(device)).cpu()
            zs.append(z)
            ys.append(y)

        zdata = torch.cat(zs, )
        ydata = torch.cat(ys, )
        
        if save_data:
            datas = TensorDataset(zdata, ydata) 
            torch.save(datas, './autoencoded_data.pt')
        
        return TensorDataset(zdata, ydata)
    
    def forward(self, x):
        e = self.model(x)

        return e

    def training_step(self, sample, batch_idx):
        x = self.model.encoder(sample[0])
        z_hat = self.model.decoder(x)

        loss = self._loss_fn(sample[0], z_hat)

        self.log("train_loss", loss.item(), prog_bar=True)

        return loss

    def validation_step(self, sample, batch_idx):
        x = self.model.encoder(sample[0])
        z_hat = self.model.decoder(x)

        val_loss = self._loss_fn(sample[0], z_hat)

        self.log("val_loss", val_loss.item(), prog_bar=True)

        return val_loss

    def configure_optimizers(self):
        opt_model = Adam([{"params": self.model_params}], lr=self.lr)
        return opt_model

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            timeout=30,
        )

        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            timeout=30,
        )

        return val_dataloader
