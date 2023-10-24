import torch
import torch.nn as nn

import pytorch_lightning as pl

import numpy as np

class SimpleUNetModules(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 16
        self.sigma = 25.0
        self.euler_maruyam_num_steps = 500
        
        
    
    
    def _marginal_prob_std(self,time_step, sigma) -> torch.Tensor:
        """Compute the mean and standard deviation of p_{0t}(x(t) | x(0)).
            A function that gives the standard deviation of the perturbation kernel

        Args:
            time_step (_type_): vector of time step
            sigma (_type_): sigma in SDE

        Returns:
            torch.Tensor: standard deviation
        """
        
        t = torch.Tensor(time_step, self.device)
        std = torch.sqrt((sigma**(2*t) - 1.) / 2. / np.log(sigma))
        
        return std

    def _diffusion_coeff(self, time_step, sigma) -> torch.Tensor:
        """Compute the diffusion coefficient of our SDE.

        Args:
            time_step (_type_): Time step ector
            sigma (_type_): The sigma in our SDE.

        Returns:
            torch.Tensor: The vector of diffusion coefficients.
        """
        
        return torch.tensor(sigma**time_step, device=self.device)
    
    def _loss_fn(self, sample:torch.Tensor, eps=1e-5):
        """The loss function for training score-based generative models.

        Args:
            sample (_type_): A mini-batch of training data
            eps (_type_, optional): A tolerance value for numerical stability. Defaults to 1e-5.
        """
        
        # Sample time uniformly from 0 to 1
        random_t = torch.rand(sample.shape[0], device=sample.device) * (1. - eps) + eps
        # Fine the noise std at the time t
        std = self._marginal_prob_std(random_t, self.sigma)
        z = torch.randn_like(sample)# get normally distributed noise
        perturbed_x = sample + std[:,None,None,None]*z
        
        # TODO evalutate model, check if the model is proper
        score = self.model(perturbed_x, random_t)
        loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
        return loss
    
    def _euler_maruyam_sampler(self,eps=1e-3, y=None, x_shape=(1,28,28)):
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
        
        t = torch.ones(self.batch_size, device=self.device)
        init_x = torch.randn(self.batch_size, *x_shape, device=self.device) * self._marginal_prob_std(t)[:,None,None,None]
        time_steps = torch.linspace(1., eps, self.euler_maruyam_num_steps, device=self.device)
        step_size = time_steps[0] - time_steps[1]
        x = init_x
        with torch.no_grad():
            for time_step in range(time_steps):
                batch_time_step = torch.ones(self.batch_size, device=self.device) * time_step
                g = self._diffusion_coeff(batch_time_step)
                mean_x = x + (g**2)[:, None, None, None] * self.model(x, batch_time_step, y=y) * step_size
                x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
        
        return mean_x
    
    def forward(self, x):
        pass
    
    def training_step(self, sample, batch_idx, optimizer_idx):
        pass
    
    def validation_step(self, sample, batch_idx):
        pass
    
    def configure_optimizers(self):
        return super().configure_optimizers()
    
    def train_dataloader(self):
        return super().train_dataloader()
    
    def val_dataloader(self):
        return super().val_dataloader()