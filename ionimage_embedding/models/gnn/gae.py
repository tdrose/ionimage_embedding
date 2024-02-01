from turtle import pos
from typing import Optional, Final

from torch_geometric.nn.models import GAE
from torch.nn import Module
import torch.nn.functional as functional
import torch

EPS: Final[float] = 1e-15

class colocGAE(GAE):

    def __init__(self, encoder: Module, decoder: Optional[Module] = None):
        super().__init__(encoder, decoder)
        self.mse = torch.nn.MSELoss()

    def recon_loss(self, z: torch.Tensor, pos_edge_index: torch.Tensor,
                   neg_edge_index: torch.Tensor) -> torch.Tensor:
        """
        Recostruction loss for colocGAE.

        Adapted function from torch_geometric.nn.models.GAE.recon_loss.
        Requires negative edge index, which is available from the dataloader.
        """

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss
    
    def coloc_loss(self, z: torch.Tensor, pos_edge_index: torch.Tensor,
                   neg_edge_index: torch.Tensor, 
                   edge_attr: torch.Tensor, neg_edge_weights: torch.Tensor) -> torch.Tensor:
        
        # Normalize z
        zn = functional.normalize(z, p=2, dim=-1)
        pos_loss = self.mse(self.decoder(zn, pos_edge_index, sigmoid=False), edge_attr)
        neg_loss = self.mse(self.decoder(zn, neg_edge_index, sigmoid=False), neg_edge_weights)

        return pos_loss + neg_loss
