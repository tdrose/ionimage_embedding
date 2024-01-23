import re
from typing import Optional, Literal, List, Union, Final

import torch
import torch.nn.functional as functional
import lightning.pytorch as pl

from .cnnClust import CNNClust
from .resnet_wrapper import ResNetWrapper
from .vit_b_16_wrapper import VitB16Wrapper
from .cae import CAE
from .pseudo_labeling import pseudo_labeling, compute_dataset_ublb


MIN_DATASETS: Final[int] = 3
NEGATIVE_SAMPLES: Final[int] = 10
TEMPERATURE: Final[float] = 0.1

class infoNCEModel(pl.LightningModule):

    clust: Union[CNNClust, ResNetWrapper, VitB16Wrapper]

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
                 architecture: Literal['cnnclust', 'vit_b_16', 'resnet18', 'resnet34', 'resnet50', 
                                       'resnet101', 'resnet152'] = 'cnnclust',
                 resnet_pretrained: bool = False,
                 vitb16_pretrained: Optional[Literal['IMAGENET1K_V1', 'IMAGENET1K_SWAG_E2E_V1', 
                                                     'IMAGENET1K_SWAG_LINEAR_V1']] = None,
                 cnn_dropout=0.1, weight_decay=1e-4,
                 clip_gradients: Optional[float] = None
                ):
        
        super(infoNCEModel, self).__init__()
        
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

    def loss_mask(self, features, uu, ll, train_datasets, index, train_images, raw_images):

        # Compute cosine between features
        features = functional.normalize(features, p=2, dim=-1)
        sim_mat = torch.matmul(features, torch.transpose(features, 0, 1))

        # Compute cosine between features and raw images
        raw_images = raw_images.reshape(raw_images.shape[0], -1)
        raw_images = functional.normalize(raw_images, p=2, dim=-1)
        raw_sim_mat = torch.matmul(features, torch.transpose(raw_images, 0, 1))

        # Compute cosine between raw images
        rr_sim_mat = torch.matmul(raw_images, torch.transpose(raw_images, 0, 1))

        # Initialize loss variable
        loss = torch.tensor(.0, requires_grad=True, device=self.device)

        counter = torch.tensor(.0, device=self.device)  

        # Dataset internal loss terms
        for i in range(len(features)):
            # Mask for features from the same dataset
            ds_mask = train_datasets == train_datasets[i]
            if sum(ds_mask) > MIN_DATASETS:
                # Compute InfoNCE loss
                pos = torch.exp(raw_sim_mat[i, i]/TEMPERATURE)

                # Pick the lowest NEGATIVE_SAMPLES values from the same dataset based on rr_sim_mat
                neg_mask = torch.topk(rr_sim_mat[i, ds_mask], 
                                      min([NEGATIVE_SAMPLES, sum(ds_mask)]), 
                                      largest=False).indices
                
                neg = torch.exp(raw_sim_mat[i, ds_mask][neg_mask]/TEMPERATURE).sum()

                # Compute loss
                loss = loss - torch.log(pos/neg)
                counter += 1.
        
        # Same ion loss terms
        ion_submat = self.ion_label_mat[index, :][:, index]
        for i in range(len(features)):
            if sum(ion_submat[i, :]) > 0:
                others = torch.arange(len(features), device=self.device)[ion_submat[i, :] == 1]
                for other in others:
                    # Compute InfoNCE loss
                    pos = torch.exp(sim_mat[i, other]/TEMPERATURE)

                    # Pick the lowest NEGATIVE_SAMPLES values from the same dataset based on rr_sim_mat
                    ds_mask1 = train_datasets == train_datasets[i]
                    ds_mask2 = train_datasets == train_datasets[other]
                    if sum(ds_mask1) > MIN_DATASETS and sum(ds_mask2) > MIN_DATASETS:

                        neg_mask1 = torch.topk(rr_sim_mat[i, ds_mask1],
                                               min([NEGATIVE_SAMPLES, sum(ds_mask1)]),
                                               largest=False).indices
                        neg_mask2 = torch.topk(rr_sim_mat[i, ds_mask2],
                                               min([NEGATIVE_SAMPLES, sum(ds_mask2)]),
                                               largest=False).indices
                        
                        neg1 = torch.exp(raw_sim_mat[i, ds_mask1][neg_mask1]/TEMPERATURE).sum()
                        neg2 = torch.exp(raw_sim_mat[i, ds_mask2][neg_mask2]/TEMPERATURE).sum()
                        
                        loss = loss  - torch.log(pos/(neg1 + neg2))
                        counter += 1.

        
        # Return mean loss
        loss = loss/counter
        return loss

    def contrastive_loss(self, features, uu, ll, train_datasets, index, train_images, raw_images):
        
        return self.loss_mask(features, uu, ll, train_datasets, 
                              index, train_images, raw_images=raw_images)
    
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
        
        train_x, index, train_datasets, train_ions, untransformed_images = batch

        
        self.knn_adj = self.knn_adj.to(self.device)
        self.ion_label_mat = self.ion_label_mat.to(self.device)
        
        train_datasets = train_datasets.reshape(-1)
        train_ions = train_ions.reshape(-1)
        
        if self.cae is None:
            features, x_p = self.forward(train_x)
            uti_features, uti_x_p = self.forward(untransformed_images)
            loss = self.contrastive_loss(features=features, uu=self.curr_upper, ll=self.curr_lower, 
                                         train_datasets=train_datasets, index=index, 
                                         train_images=train_x, raw_images=uti_features)
            self.log('Training loss', loss, on_step=False, on_epoch=True, 
                     logger=True, prog_bar=True)
            return loss
        
        else:
            features, x_p = self.forward(train_x)
            uti_features, uti_x_p = self.forward(untransformed_images)
            loss_cae = self.mse_loss(x_p, train_x)
            loss_clust = self.contrastive_loss(features=features, uu=self.curr_upper, 
                                               ll=self.curr_lower, 
                                               train_datasets=train_datasets, index=index, 
                                               train_images=train_x, 
                                               raw_images=uti_features)
            loss = loss_cae + loss_clust
            self.log('Training loss', loss, on_step=False, on_epoch=True, 
                     logger=True, prog_bar=True)
            self.log('Training CAE-loss', loss_cae, on_step=False, on_epoch=True, 
                     logger=True, prog_bar=True)
            self.log('Training CLR-loss', loss_clust, on_step=False, on_epoch=True, 
                     logger=True, prog_bar=True)
            
            return loss
    
    def validation_step(self, batch, batch_idx):
        val_x, index, val_datasets, val_ions, untransformed_images = batch
        
        self.knn_adj = self.knn_adj.to(self.device)
        self.ion_label_mat = self.ion_label_mat.to(self.device)
        
        val_datasets = val_datasets.reshape(-1)
        val_ions = val_ions.reshape(-1)

        if self.cae is None:
            features, x_p = self.forward(val_x)
            uti_features, uti_x_p = self.forward(untransformed_images)
            loss = self.contrastive_loss(features=features, uu=self.curr_upper, ll=self.curr_lower, 
                                         train_datasets=val_datasets, index=index, 
                                         train_images=val_x, raw_images=uti_features)
            self.log('Validation loss', loss, on_step=False, on_epoch=True, 
                     logger=True, prog_bar=True)

            return loss
        
        else:
            features, x_p = self.forward(val_x)
            uti_features, uti_x_p = self.forward(untransformed_images)
            loss_cae = self.mse_loss(x_p, val_x)
            loss_clust = self.contrastive_loss(features=features, uu=self.curr_upper, 
                                               ll=self.curr_lower, 
                                               train_datasets=val_datasets, index=index, 
                                               train_images=val_x, raw_images=uti_features)
            loss = loss_cae + loss_clust
            self.log('Validation loss', loss, on_step=False, on_epoch=True, 
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
