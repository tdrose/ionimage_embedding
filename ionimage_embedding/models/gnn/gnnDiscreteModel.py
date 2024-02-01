from typing import Any, Literal, Optional

import lightning.pytorch as pl
import torch
from torch_geometric.nn import GCNConv

from .gae import colocGAE

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
    

class gnnDiscreteModel(pl.LightningModule):

    def __init__(self, n_ions: int, latent_dims: int=10,
                  encoding: Literal['onehot', 'learned']= 'onehot',
                  embedding_dims: int=10, lr=1e-3
                  ):     
        super(gnnDiscreteModel, self).__init__()

        self.n_ions = n_ions
        self.encoding: Literal['onehot', 'learned'] = encoding
        self.latent_dims = latent_dims

        self.lr = lr

        # Create encoding
        if self.encoding == 'onehot':
            self.embedding = self.onehot
            self.embedding_dims = self.n_ions
        elif self.encoding == 'learned':
            self.embedding = torch.nn.Embedding(self.n_ions, embedding_dims, 
                                                device=self.device)
            self.embedding_dims = embedding_dims
        else:
            raise ValueError('encoding must be one of "onehot" or "learned"')
        
        # Set GAE model
        self.gae = colocGAE(GCNEncoder(in_channels=self.embedding_dims, 
                                       hidden_channels=self.latent_dims*2, 
                                       out_channels=self.latent_dims))

    def onehot(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(x, num_classes=self.n_ions).float().to(self.device)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        return self.gae(x, edge_index)
    
    def training_step(self, batch, batch_idx):
        z = self.forward(batch.x, batch.edge_index)
        
        loss = self.gae.recon_loss(z, batch.edge_index, batch.neg_edge_index)
        
        self.log('Training loss', loss, 
                 on_step=False, on_epoch=True, logger=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        z = self.forward(batch.x, batch.edge_index)
        
        loss = self.gae.recon_loss(z, batch.edge_index, batch.neg_edge_index)
        
        self.log('Validation loss', loss, 
                 on_step=False, on_epoch=True, logger=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
