from typing import Literal

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATv2Conv, GraphConv

from .gae import colocGAE
from ..constants import TRAINING_LOSS, VALIDATION_LOSS

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers: int=2,
                 activation: Literal['softmax', 'relu', 'sigmoid', 'none']='none',
                 gnn_layer_type: Literal['GCNConv', 'GATv2Conv', 'GraphConv'] = 'GCNConv'):
        super().__init__()

        if num_layers < 1:
            raise ValueError('num_layers must be greater than 1')
        
        self.edge_attr_reshaping = False
        
        if gnn_layer_type in ['GATv2Conv']:
            self.edge_attr_reshaping = True

        self.convs = nn.ModuleList()

        if gnn_layer_type == 'GCNConv':
            # Define the first GCNConv layer
            self.conv1 = GCNConv(in_channels, hidden_channels)
            for _ in range(num_layers-1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
        elif gnn_layer_type == 'GATv2Conv':
            self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=1, edge_dim=1)
            for _ in range(num_layers-1):
                self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=1, edge_dim=1))
        elif gnn_layer_type == 'GraphConv':
            self.conv1 = GraphConv(in_channels, hidden_channels, aggr='mean')
            for _ in range(num_layers-1):
                self.convs.append(GraphConv(hidden_channels, hidden_channels, aggr='mean'))
        else:
            raise ValueError('gnn_layer_type must be one of "GCNConv" or "GATv2Conv"')


        # Define the output projection layer
        self.proj = nn.Linear(hidden_channels, out_channels)

        # Previous implementation
        # self.conv1 = GCNConv(in_channels, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, out_channels)

        if activation == 'softmax':
            self.activation = torch.nn.Softmax(dim=-1)
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation == 'none':
            self.activation = lambda x: x
        else:
            raise ValueError('activation must be one of "softmax", "relu", "sigmoid", or "none"')

    def forward(self, x, edge_index, edge_attr):
        
        if self.edge_attr_reshaping:
            edge_attr = edge_attr.unsqueeze(-1)
        
        x = self.conv1(x, edge_index, edge_attr).relu()

        # Apply intermediate GCNConv layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr).relu()

        # Apply the output projection layer
        x = self.proj(x)
        
        return self.activation(x)
    

class gnnDiscreteModel(pl.LightningModule):

    embedding_dims: int

    def __init__(self, n_ions: int, latent_dims: int=10,
                  encoding: Literal['onehot', 'learned']= 'onehot',
                  loss: Literal['recon', 'coloc'] = 'recon',
                  activation: Literal['softmax', 'relu', 'sigmoid', 'none']='none',
                  num_layers: int=2,
                  embedding_dims: int=10, lr=1e-3,
                  gnn_layer_type: Literal['GCNConv', 'GATv2Conv', 'GraphConv'] = 'GCNConv'
                  ):     
        super(gnnDiscreteModel, self).__init__()

        self.n_ions = n_ions
        self.encoding: Literal['onehot', 'learned'] = encoding
        self.latent_dims = latent_dims
        self.loss: Literal['recon', 'coloc'] = loss
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
                                       hidden_channels=self.embedding_dims//2, 
                                       out_channels=self.latent_dims, 
                                       activation=activation, num_layers=num_layers, 
                                       gnn_layer_type=gnn_layer_type))

    def onehot(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(x, num_classes=self.n_ions).float().to(self.device)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        return self.gae(x, edge_index, edge_attr)
    
    def get_loss(self, z: torch.Tensor, edge_index: torch.Tensor, neg_edge_index: torch.Tensor,
                    edge_attr: torch.Tensor, neg_edge_weights: torch.Tensor) -> torch.Tensor:
            
            if self.loss == 'recon':
                return self.gae.recon_loss(z, edge_index, neg_edge_index)
            elif self.loss == 'coloc':
                return self.gae.coloc_loss(z, edge_index, neg_edge_index, 
                                           edge_attr, neg_edge_weights)
            else:
                raise ValueError('loss must be one of "recon" or "coloc"')
    
    def training_step(self, batch, batch_idx):
        z = self.forward(batch.x, batch.edge_index, batch.edge_attr)
        
        loss = self.get_loss(z, batch.edge_index, batch.neg_edge_index, 
                             batch.edge_attr, batch.neg_edge_weights)
        
        self.log(TRAINING_LOSS, loss, 
                 on_step=False, on_epoch=True, logger=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        z = self.forward(batch.x, batch.edge_index, batch.edge_attr)
        
        loss = self.get_loss(z, batch.edge_index, batch.neg_edge_index, 
                             batch.edge_attr, batch.neg_edge_weights)
        
        self.log(VALIDATION_LOSS, loss, 
                 on_step=False, on_epoch=True, logger=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
