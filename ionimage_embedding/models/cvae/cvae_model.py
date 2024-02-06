from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as functional
import lightning.pytorch as pl

from .vanilla import Encoder, Decoder
from ..constants import TRAINING_LOSS, VALIDATION_LOSS


class CVAEmodel(pl.LightningModule):
    def __init__(self, height: int, width: int, n_classes: int, latent_dim: int=7, 
                 activation: Literal['softmax', 'relu', 'sigmoid'] = 'softmax',
                 lr: float=0.01, weight_decay: float=1e-4):
        
        super(CVAEmodel, self).__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.latent_dim = latent_dim
        self.height = height
        self.width = width
        self.n_classes = n_classes

        self.encoder = Encoder(height=height, width=width, n_classes=n_classes, 
                               encoder_dim=latent_dim*2, activation=activation)
        
        self.decoder = Decoder(height=height, width=width, n_classes=n_classes, 
                               encoder_dim=latent_dim, activation=activation,
                               encoder_l1height=self.encoder.l1height, 
                               encoder_l1width=self.encoder.l1width, 
                               encoder_l2height=self.encoder.l2height,
                               encoder_l2width=self.encoder.l2width)
        
        self.linear1 = nn.Linear(latent_dim*2, latent_dim)
        self.linear2 = nn.Linear(latent_dim*2, latent_dim)
        
        self.mse_loss = torch.nn.MSELoss()

        self.N = torch.distributions.Normal(0, 1)
        # Get sampling working on GPU
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()

    def forward(self, x, c, ion_labels):
        
        # One hot encoding of condition
        nc = functional.one_hot(c, num_classes=self.n_classes).to(self.device)
        nc = nc.reshape((nc.shape[0], -1))
        # Encode
        z = self.encoder(x, nc)
        
        # Reparameterize
        mu, sigma = self.reparameterize(z)
        
        # Sample latent space
        zs = self.sample(mu, sigma)
        
        # compute KL divergence
        kl_div = self.kl_divergence(mu, sigma)

        # Compute same ion loss
        same_ion_loss = self.same_ion_loss(zs, ion_labels)

        # Decode
        x_hat = self.decoder(zs, nc)

        # Reconstruction loss
        recon_loss = self.reconstruction_loss(x, x_hat) 

        return x_hat, zs, recon_loss, kl_div, same_ion_loss

    def reparameterize(self, z):
        mu = self.linear1(z)
        sigma = torch.exp(self.linear2(z))

        return mu, sigma

    def reconstruction_loss(self, x, x_hat):
        return self.mse_loss(x, x_hat)
    
    def sample(self, mu, sigma):
        return mu+sigma*self.N.sample(mu.shape).to(self.device)

    def kl_divergence(self, mu, sigma):
        return (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

    def same_ion_loss(self, z, ion_labels):
        ion_labels = ion_labels.reshape((-1,))
        
        unique_labels = torch.unique(ion_labels, sorted=True).detach().cpu().numpy()
        
        same_ion_loss = torch.tensor(0.0, device=self.device)

        for unique_label in unique_labels:
            mask = ion_labels == unique_label
            masked_z = z[mask]
            if masked_z.shape[0] > 1:
                same_ion_loss += torch.cdist(masked_z, masked_z).mean()

        return same_ion_loss
    

    def training_step(self, batch, batch_idx):

        train_x, index, train_datasets, train_ions, untransformed_images = batch

        x_hat, z, recon_loss, kl_div, same_ion_loss = self.forward(train_x, 
                                                                   train_datasets, 
                                                                   train_ions)
        
        loss = recon_loss + kl_div + same_ion_loss

        self.log(TRAINING_LOSS, loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
            
        val_x, index, val_datasets, val_ions, untransformed_images = batch

        x_hat, z, recon_loss, kl_div, same_ion_loss = self.forward(val_x, 
                                                                   val_datasets, 
                                                                   val_ions)
        
        loss = recon_loss + kl_div + same_ion_loss

        self.log(VALIDATION_LOSS, loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        model_params = self.parameters()
        return torch.optim.RMSprop(params=model_params, lr=self.lr, weight_decay=self.weight_decay)
