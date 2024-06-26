from typing import Optional, Literal, List, Union
from sympy import im

import torch
import torch.nn as nn
import torch.nn.functional as functional
import lightning.pytorch as pl

from .cnnClust import CNNClust
from .resnet_wrapper import ResNetWrapper
from .vit_b_16_wrapper import VitB16Wrapper
from .cae import CAE
from .pseudo_labeling import compute_dataset_ublb
from ...coloc.utils import torch_cosine
from ..constants import TRAINING_LOSS, VALIDATION_LOSS

class colocContrastModel(pl.LightningModule):

    clust: Union[CNNClust, ResNetWrapper, VitB16Wrapper]

    def __init__(self, 
                 height, 
                 width,
                 num_cluster,
                 ion_label_mat: torch.Tensor,
                 knn_adj: Optional[torch.Tensor],
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
                 architecture: Literal['cnnclust', 'vit_b_16', 'resnet18', 'resnet34', 'resnet50', 
                                       'resnet101', 'resnet152'] = 'cnnclust',
                 resnet_pretrained: bool = False,
                 vitb16_pretrained: Optional[Literal['IMAGENET1K_V1', 'IMAGENET1K_SWAG_E2E_V1', 
                                                     'IMAGENET1K_SWAG_LINEAR_V1']] = None,
                 cnn_dropout=0.1, weight_decay=1e-4,
                 clip_gradients: Optional[float] = None
                ):
        
        super(colocContrastModel, self).__init__()
        
        # Model sizes
        self.height = height
        self.width = width
        self.cae_encoder_dim = encoder_dim
        self.num_cluster = num_cluster
        
        # KNN
        self.KNN = knn
        self.knn_adj: Optional[torch.Tensor] = knn_adj
        
        # Trainig related
        self.weight_decay = weight_decay
        self.lr = lr
        self.clip_grads = clip_gradients
        # self.overweight_cae = overweight_cae
        self.mse_loss = torch.nn.MSELoss()
        self.bceloss = nn.BCELoss()
        
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
        
        if architecture == 'cnnclust':
            self.clust = CNNClust(num_clust=self.num_cluster, height=self.height, width=self.width, 
                                activation=activation, dropout=cnn_dropout)
        elif architecture == 'vit_b_16':
            self.clust = VitB16Wrapper(num_clust=self.num_cluster, activation=activation, 
                                       pretrained=vitb16_pretrained, 
                                       height=self.height, width=self.width)
        elif architecture in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            self.clust = ResNetWrapper(num_clust=self.num_cluster, activation=activation, 
                                       resnet=architecture, 
                                       pretrained=resnet_pretrained, height=self.height, 
                                       width=self.width)
        
    def cl(self, neg_loc, pos_loc, sim_mat):
        pos = sim_mat[pos_loc==1.]
        neg = sim_mat[neg_loc==1.]
        out = torch.clip(torch.cat([pos, neg]), 0.0, 1.0)
        
        target = torch.cat([torch.ones(pos.shape[0], device=self.device), 
                            torch.zeros(neg.shape[0], device=self.device)])

        return self.bceloss(out, target)
    
    def compute_ublb(self, features):
        
        features = functional.normalize(features, p=2, dim=-1)

        sim_mat = torch.matmul(features, torch.transpose(features, 0, 1))

        # mask = torch.eye(sim_mat.size(0), dtype=torch.bool)
        # asked_matrix = sim_mat[~mask]
        # ub = torch.quantile(masked_matrix, uu/100).detach()
        # lb = torch.quantile(masked_matrix, ll/100).detach()

        return sim_mat

    def loss_mask(self, features, uu, ll, train_datasets, index, train_images, raw_images):

        # Model representation similarities
        sim_mat = self.compute_ublb(features)
        
        # Compute cosine between all input images (Using un-transformed images)
        gt_cosine = torch_cosine(raw_images.reshape(raw_images.shape[0], -1))
        gt_cosine = gt_cosine.to(self.device)

        # Calculate dataset ub and lb, but on the GROUND TRUTH COSINE
        dataset_ub, dataset_lb = compute_dataset_ublb(gt_cosine, ds_labels=train_datasets,
                                                      lower_bound=ll, upper_bound=uu, 
                                                      device=self.device)

        # print(dataset_ub)
        # print(dataset_lb)

        ub_m = torch.ones(sim_mat.shape, device=self.device)
        lb_m = torch.zeros(sim_mat.shape, device=self.device)

        # Loop over all datasets
        for ds in torch.unique(train_datasets):
            ds_v = train_datasets == ds
            # Mask to subset similarities just to one dataset
            mask = torch.outer(ds_v, ds_v)

            # Create dataset specific threshold matrices
            ub_m[mask] = dataset_ub[ds]
            lb_m[mask] = dataset_lb[ds]
        
        # Apply thresholds to GROUND TRUGH COSINE 
        pos_loc = (gt_cosine > ub_m).float()
        neg_loc = (gt_cosine < lb_m).float()


        # Align the same ions
        ion_submat = self.ion_label_mat[index, :][:, index]
        
        pos_loc = torch.maximum(pos_loc, ion_submat)
        neg_loc = torch.minimum(neg_loc, 1 - ion_submat)

        # Remove diagonal
        mask = torch.eye(pos_loc.size(0), dtype=torch.bool)
        pos_loc[mask] = 0.
        neg_loc[mask] = 0.

        return pos_loc, neg_loc, sim_mat

    def contrastive_loss(self, features, uu, ll, train_datasets, index, train_images, raw_images):
        
        pos_loc, neg_loc, sim_mat = self.loss_mask(features, uu, ll, train_datasets, 
                                                   index, train_images, 
                                                   raw_images=raw_images)

        return self.cl(neg_loc, pos_loc, sim_mat)
    
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
        
        train_x, index, train_datasets, train_ions, untransformed_images, _ = batch
        
        # self.knn_adj = self.knn_adj.to(self.device)
        self.ion_label_mat = self.ion_label_mat.to(self.device)
        
        train_datasets = train_datasets.reshape(-1)
        train_ions = train_ions.reshape(-1)
        
        if self.cae is None:
            features, x_p = self.forward(train_x)
            loss = self.contrastive_loss(features=features, uu=self.curr_upper, 
                                         ll=self.curr_lower, train_datasets=train_datasets, 
                                         index=index, train_images=train_x, 
                                         raw_images=untransformed_images)
            self.log(TRAINING_LOSS, loss, on_step=False, on_epoch=True, 
                     logger=True, prog_bar=True)
            return loss
        
        else:
            features, x_p = self.forward(train_x)
            loss_cae = self.mse_loss(x_p, train_x)
            loss_clust = self.contrastive_loss(features=features, uu=self.curr_upper, 
                                               ll=self.curr_lower, train_datasets=train_datasets, 
                                               index=index, train_images=train_x, 
                                               raw_images=untransformed_images)
            loss = loss_cae + loss_clust
            self.log(TRAINING_LOSS, loss, on_step=False, on_epoch=True, 
                     logger=True, prog_bar=True)
            self.log('Training CAE-loss', loss_cae, on_step=False, on_epoch=True, 
                     logger=True, prog_bar=True)
            self.log('Training CLR-loss', loss_clust, on_step=False, on_epoch=True, 
                     logger=True, prog_bar=True)
            
            return loss
    
    def validation_step(self, batch, batch_idx):
        val_x, index, val_datasets, val_ions, untransformed_images, _ = batch
        
        # self.knn_adj = self.knn_adj.to(self.device)
        self.ion_label_mat = self.ion_label_mat.to(self.device)
        
        val_datasets = val_datasets.reshape(-1)
        val_ions = val_ions.reshape(-1)

        if self.cae is None:
            features, x_p = self.forward(val_x)
            loss = self.contrastive_loss(features=features, uu=self.curr_upper, 
                                         ll=self.curr_lower, train_datasets=val_datasets, 
                                         index=index, train_images=val_x, 
                                         raw_images=untransformed_images)
            self.log(VALIDATION_LOSS, loss, on_step=False, on_epoch=True, 
                     logger=True, prog_bar=True)

            return loss
        
        else:
            features, x_p = self.forward(val_x)
            loss_cae = self.mse_loss(x_p, val_x)
            loss_clust = self.contrastive_loss(features=features, uu=self.curr_upper, 
                                               ll=self.curr_lower, 
                                               train_datasets=val_datasets, index=index, 
                                               train_images=val_x, 
                                               raw_images=untransformed_images)
            loss = loss_cae + loss_clust
            self.log(VALIDATION_LOSS, loss, on_step=False, on_epoch=True, 
                     logger=True, prog_bar=True)
            self.log('Validation CAE-loss', loss_cae, on_step=False, on_epoch=True, 
                     logger=True, prog_bar=True)
            self.log('Validation CLR-loss', loss_clust, on_step=False, on_epoch=True, 
                     logger=True, prog_bar=True)
            
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
