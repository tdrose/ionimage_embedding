from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as functional
import lightning.pytorch as pl

from .cnnClust import CNNClust
from .cae import CAE
from .pseudo_labeling import pseudo_labeling, compute_dataset_ublb
from ..coloc.utils import torch_cosine

class regContrastModel(pl.LightningModule):
    def __init__(self, 
                 height, 
                 width,
                 num_cluster,
                 ion_label_mat: torch.Tensor,
                 knn_adj: torch.Tensor,
                 activation: Literal['softmax', 'relu', 'sigmoid']='softmax',
                 encoder_dim=7,
                 initial_upper: float = 98.,
                 initial_lower: float = 46.,
                 upper_iteration: float = 1,
                 lower_iteration: float = 1,
                 dataset_specific_percentiles: bool = False,
                 lr=0.01,
                 cae_pretrained_model=None,
                 knn=False,
                 cnn_dropout=0.1, weight_decay=1e-4,
                 clip_gradients: Optional[float] = None
                ):
        
        super(regContrastModel, self).__init__()
        
        # Model sizes
        self.height = height
        self.width = width
        self.cae_encoder_dim = encoder_dim
        self.num_cluster = num_cluster
        
        # KNN
        self.KNN = knn
        self.knn_adj = knn_adj
        
        # Trainig related
        self.weight_decay = weight_decay
        self.lr = lr
        self.clip_grads = clip_gradients
        # self.overweight_cae = overweight_cae
        self.mse_loss = torch.nn.MSELoss()
        self.mseloss = torch.nn.MSELoss()
        
        # Pseudo labeling parameters
        self.initial_upper = initial_upper
        self.initial_lower = initial_lower
        self.upper_iteration = upper_iteration
        self.lower_iteration = lower_iteration
        self.dataset_specific_percentiles = dataset_specific_percentiles
        self.curr_lower = initial_lower
        self.curr_upper = initial_upper
        
        self.ion_label_mat = ion_label_mat
        
        if cae_pretrained_model is None:
            self.cae = CAE(height=self.height, width=self.width, encoder_dim=self.cae_encoder_dim)
        elif cae_pretrained_model is False:
            self.cae = None
        else:
            self.cae = cae_pretrained_model
        self.clust = CNNClust(num_clust=self.num_cluster, height=self.height, width=self.width, activation=activation, dropout=cnn_dropout)

    def cl(self, ds_mask, sim_mat, gt_cosine):
        # pos_entropy = torch.mul(-torch.log(torch.clip(sim_mat, 1e-10, 1)), pos_loc)
        # neg_entropy = torch.mul(-torch.log(torch.clip(1 - sim_mat, 1e-10, 1)), neg_loc)
        #pos_entropy = -torch.log(sim_mat[pos_loc==1.])
        #neg_entropy = -torch.log(sim_mat[neg_loc==1.])
        # print('cl')
        # print(pos_entropy)
        # print(neg_entropy)
        # CNN loss
        #contrastive_loss = pos_entropy.sum() / pos_loc.sum() + neg_entropy.sum() / neg_loc.sum()
        # print(contrastive_loss)
        # print()
        #return contrastive_loss
        out = sim_mat[ds_mask==1.]
        target = gt_cosine[ds_mask==1.]

        return self.mseloss(out, target)
    
    def compute_ublb(self, features):
        
        features = functional.normalize(features, p=2, dim=-1)
        # Second normalization not necessary. Keeping it here for reference, will be removed in the future.
        # features = features / features.norm(dim=1)[:, None]

        sim_mat = torch.matmul(features, torch.transpose(features, 0, 1))

        # mask = torch.eye(sim_mat.size(0), dtype=torch.bool)
        # asked_matrix = sim_mat[~mask]
        # ub = torch.quantile(masked_matrix, uu/100).detach()
        # lb = torch.quantile(masked_matrix, ll/100).detach()

        return sim_mat
    
    def loss_mask(self, features, uu, ll, train_datasets, index, train_images):

        # Model representation similarities
        sim_mat = self.compute_ublb(features)
        
        # Compute cosine between all input images
        gt_cosine = torch_cosine(train_images.reshape(train_images.shape[0], -1))
        gt_cosine = gt_cosine.to(self.device)

        # Only for those values the loss will be evaluated
        ds_mask = torch.zeros(sim_mat.shape, device=self.device)

        # Loop over all datasets
        for ds in torch.unique(train_datasets):
            ds_v = train_datasets == ds
            
            # Mask to subset similarities just to one dataset
            mask = torch.outer(ds_v, ds_v)
            mask2 = torch.eye(ds_mask.size(0), dtype=torch.bool)
            mask[mask2] = 0.

            # Set maskin with datasets to 1
            ds_mask[mask] = 1

        # Align the same ions
        ion_submat = self.ion_label_mat[index, :][:, index]
        
        # Set same ions to 1 in target
        ds_mask = torch.maximum(ds_mask, ion_submat)

        gt_cosine[ion_submat==1.] = 1
        
        return ds_mask, sim_mat, gt_cosine
    

    def contrastive_loss(self, features, uu, ll, train_datasets, index, train_images):
        
        ds_mask, sim_mat, gt_cosine = self.loss_mask(features, uu, ll, train_datasets, index, train_images)

        return self.cl(ds_mask, sim_mat, gt_cosine)
    
    def forward(self, x):
        if self.cae is None:
            features = self.clust(x)
            return features, x
        else:
            x_p = self.cae(x)
            features = self.clust(x_p)
            return features, x_p
        
    def embed_layers(self, x):
        if self.cae is None:
            features = self.clust.embed_layers(x)
            return features, x
        else:
            x_p = self.cae(x)
            features = self.clust.embed_layers(x_p)
            return features, x_p
    
    def training_step(self, batch, batch_idx):
        
        train_x, index, train_datasets, train_ions = batch
        
        self.knn_adj = self.knn_adj.to(self.device)
        self.ion_label_mat = self.ion_label_mat.to(self.device)
        
        train_datasets = train_datasets.reshape(-1)
        train_ions = train_ions.reshape(-1)
        
        if self.cae is None:
            features, x_p = self.forward(train_x)
            loss = self.contrastive_loss(features=features, uu=self.curr_upper, ll=self.curr_lower, train_datasets=train_datasets, 
                                         index=index, train_images=train_x)
            self.log('Training loss', loss, on_step=False, on_epoch=True, logger=False, prog_bar=True)
            return loss
        
        else:
            features, x_p = self.forward(train_x)
            loss_cae = self.mse_loss(x_p, train_x)
            loss_clust = self.contrastive_loss(features=features, uu=self.curr_upper, ll=self.curr_lower, train_datasets=train_datasets, 
                                               index=index, train_images=train_x)
            loss = loss_cae + loss_clust
            self.log('Training loss', loss, on_step=False, on_epoch=True, logger=False, prog_bar=True)
            self.log('Training CAE-loss', loss_cae, on_step=False, on_epoch=True, logger=False, prog_bar=True)
            self.log('Training CLR-loss', loss_clust, on_step=False, on_epoch=True, logger=False, prog_bar=True)
            
            return loss
    
    def validation_step(self, batch, batch_idx):
        val_x, index, val_datasets, val_ions = batch
        
        self.knn_adj = self.knn_adj.to(self.device)
        self.ion_label_mat = self.ion_label_mat.to(self.device)
        
        val_datasets = val_datasets.reshape(-1)
        val_ions = val_ions.reshape(-1)

        if self.cae is None:
            features, x_p = self.forward(val_x)
            loss = self.contrastive_loss(features=features, uu=self.curr_upper, ll=self.curr_lower, train_datasets=val_datasets, 
                                         index=index, train_images=val_x)
            self.log('Validation loss', loss, on_step=False, on_epoch=True, logger=False, prog_bar=True)

            return loss
        
        else:
            features, x_p = self.forward(val_x)
            loss_cae = self.mse_loss(x_p, val_x)
            loss_clust = self.contrastive_loss(features=features, uu=self.curr_upper, ll=self.curr_lower, train_datasets=val_datasets, index=index)
            loss = loss_cae + loss_clust
            self.log('Validation loss', loss, on_step=False, on_epoch=True, logger=False, prog_bar=True)
            self.log('Validation CAE-loss', loss_cae, on_step=False, on_epoch=True, logger=False, prog_bar=True)
            self.log('Validation CLR-loss', loss_clust, on_step=False, on_epoch=True, logger=False, prog_bar=True)
            
            return loss
    
    def on_train_epoch_end(self, *args, **kwargs):
        self.curr_upper -= self.upper_iteration
        self.curr_lower += self.lower_iteration
    
    def configure_optimizers(self):
        if self.cae is None:
            model_params = self.clust.parameters()
        else:
            model_params = list(self.cae.parameters()) + list(self.clust.parameters())
        return torch.optim.RMSprop(params=model_params, lr=self.lr, weight_decay=self.weight_decay)
