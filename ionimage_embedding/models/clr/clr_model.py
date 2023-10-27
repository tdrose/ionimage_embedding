import torch
import torch.nn as nn
import torch.nn.functional as functional
import lightning.pytorch as pl

import math
from .cnnClust import CNNClust
from .pseudo_labeling import pseudo_labeling, compute_dataset_ublb


class CLRmodel(pl.LightningModule):
    def __init__(self, 
                 height, 
                 width,
                 num_cluster,
                 ion_label_mat,
                 activation='softmax',
                 encoder_dim=7,
                 initial_upper: int = 98,
                 initial_lower: int = 46,
                 upper_iteration: float = 1,
                 lower_iteration: float = 4,
                 dataset_specific_percentiles: bool = False,
                 lr=0.01,
                 cae_pretrained_model=None,
                 knn=False, knn_adj = None,
                 cnn_dropout=0.1, weight_decay=1e-4,
                 clip_gradients: float = None, overweight_cae=1000
                ):
        
        super(CLRmodel, self).__init__()
        
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
        self.clip_gradients = clip_gradients
        self.overweight_cae = overweight_cae
        self.mse_loss = torch.nn.MSELoss()
        
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
        else:
            self.cae = cae_pretrained_model
        self.clust = CNNClust(num_clust=self.num_cluster, height=self.height, width=self.width, activation=activation, dropout=cnn_dropout)
        
        
    def cl(self, neg_loc, pos_loc, sim_mat):
        pos_entropy = torch.mul(-torch.log(torch.clip(sim_mat, 1e-10, 1)), pos_loc)
        neg_entropy = torch.mul(-torch.log(torch.clip(1 - sim_mat, 1e-10, 1)), neg_loc)

        # CNN loss
        contrastive_loss = pos_entropy.sum() / pos_loc.sum() + neg_entropy.sum() / neg_loc.sum()

        return contrastive_loss
    
    def compute_ublb(self, features, uu, ll, train_datasets, index):
        features = functional.normalize(features, p=2, dim=-1)
        features = features / features.norm(dim=1)[:, None]

        sim_mat = torch.matmul(features, torch.transpose(features, 0, 1))

        mask = torch.eye(sim_mat.size(0), dtype=torch.bool)
        masked_matrix = sim_mat[~mask]

        ub = torch.quantile(masked_matrix, uu/100).detach()
        lb = torch.quantile(masked_matrix, ll/100).detach()

        return ub, lb, sim_mat

    def contrastive_loss(self, features, uu, ll, train_datasets, index):
        ub, lb, sim_mat = self.compute_ublb(features, uu, ll, train_datasets, index)
        
        dataset_ub = None
        dataset_lb = None
        if self.dataset_specific_percentiles:
            
            dataset_ub, dataset_lb = compute_dataset_ublb(sim_mat, ds_labels=train_datasets,
                                                          lower_bound=ll, upper_bound=uu)

        pos_loc, neg_loc = pseudo_labeling(ub=ub, lb=lb, sim=sim_mat, index=index, knn=self.KNN,
                                           knn_adj=self.knn_adj, ion_label_mat=self.ion_label_mat,
                                           dataset_specific_percentiles=self.dataset_specific_percentiles,
                                           dataset_ub=dataset_ub, dataset_lb=dataset_lb,
                                           ds_labels=train_datasets, device=None)


        return self.cl(neg_loc, pos_loc, sim_mat)
    
    def forward(self, x):
        x_p = self.cae(x)
        features = self.clust(x_p)
        return features, x_p
    
    def training_step(self, batch, batch_idx):
        self.curr_upper -= self.upper_iteration
        self.curr_lower += self.lower_iteration
        
        train_x, index, train_datasets, train_ions = batch
        
        train_datasets = train_datasets.reshape(-1)
        train_ions = train_ions.reshape(-1)
        
        features, x_p = self.forward(train_x)
        
        loss_cae = self.mse_loss(x_p, train_x)
        loss_clust = self.contrastive_loss(features=features, uu=self.curr_upper, ll=self.curr_lower, train_datasets=train_datasets, index=index)
        
        return self.overweight_cae*loss_cae + loss_clust
    
    def validation_step(self, batch, batch_idx):
        val_x, index, val_datasets, val_ions = batch
        
        val_datasets = val_datasets.reshape(-1)
        val_ions = val_ions.reshape(-1)
        
        features, x_p = self.forward(val_x)
        
        loss_cae = self.mse_loss(x_p, val_x)
        loss_clust = self.contrastive_loss(features=features, uu=self.curr_upper, ll=self.curr_lower, train_datasets=val_datasets, index=index)
        
        self.curr_upper -= self.upper_iteration
        self.curr_lower += self.lower_iteration

        return self.overweight_cae*loss_cae + loss_clust
    
    def configure_optimizers(self):
        model_params = list(self.cae.parameters()) + list(self.clust.parameters())
        return torch.optim.RMSprop(params=model_params, lr=self.lr, weight_decay=self.weight_decay)