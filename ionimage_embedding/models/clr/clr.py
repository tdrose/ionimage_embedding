import torch
import numpy as np
from typing import Literal
import torch.nn.functional as functional
import torchvision.transforms as transforms
import lightning.pytorch as pl

from random import sample
from sklearn import preprocessing
import math

from .cae import CAE
from .clr_model import CLRmodel
from .pseudo_labeling import pseudo_labeling, compute_dataset_ublb
from .utils import flip_images
from ...dataloader.clr_data import CLRdata

class CLR:
    def __init__(self,
                 data: CLRdata,
                 num_cluster: int = 7,
                 initial_upper: int = 98,
                 initial_lower: int = 46,
                 upper_iteration: float = 1,
                 lower_iteration: float = 4,
                 dataset_specific_percentiles: bool = False,
                 knn: bool = True,
                 lr: float = 0.01,
                 pretraining_epochs: int = 11,
                 training_epochs: int = 11,
                 cae_encoder_dim: int = 7,
                 lightning_device: str = 'gpu',
                 activation: Literal['softmax', 'relu', 'sigmoid'] = 'softmax',
                 clip_gradients: float = None,
                 overweight_cae: float = 1000,
                 cnn_dropout: float = 0.1,
                 weight_decay: float = 1e-4,
                 random_seed: int = 1234):

        # Image data
        self.data = data
        self._height = data.height
        self._width = data.width
        
        # Pseudo labeling parameters
        self.initial_upper = initial_upper
        self.initial_lower = initial_lower
        self.upper_iteration = upper_iteration
        self.lower_iteration = lower_iteration
        self.dataset_specific_percentiles = dataset_specific_percentiles
        self.KNN = knn
        
        # Pytorch parameters
        self.num_cluster = num_cluster
        self.activation = activation
        self.lr = lr
        self.pretraining_epochs = pretraining_epochs
        self.training_epochs = training_epochs
        self.mse_loss = torch.nn.MSELoss()
        self.cae_encoder_dim = cae_encoder_dim
        self.lightning_device = lightning_device
        self.random_seed = random_seed
        self.clip_gradients = clip_gradients
        self.overweight_cae = overweight_cae
        self.cnn_dropout = cnn_dropout
        self.weight_decay = weight_decay

        print(f'After {self.training_epochs} epochs, the upper bound will be: '
              f'{self.initial_upper - (self.training_epochs * self.upper_iteration)}.')
        print(f'After {self.training_epochs} epochs, the lower bound will be: '
              f'{self.initial_lower + (self.training_epochs * self.lower_iteration)}.')

        if (self.initial_lower + (self.training_epochs * self.lower_iteration)) >= \
                (self.initial_upper - (self.training_epochs * self.upper_iteration)):
            raise ValueError(f'Lower percentile will be higher than upper percentile parameter '
                             f'after {self.training_epochs} epochs.\n'
                             f'Change initial_upper, initial_lower, upper_iteration, lower_iteration, '
                             f'or training_epochs parameters.')

        self.knn_adj = data.knn_adj            
        self.ion_label_mat = data.ion_label_mat

        # Models
        self.cae = None
        self.clust = None

        self.train_dataloader = data.get_train_dataloader()
        
        self.val_dataloader = data.get_val_dataloader()

        # Placeholders for models
        self.cae = None
        self.clr = None

    def image_normalization(self, new_data: np.ndarray = None):
        return self.data.image_normalization(new_data)

    def train(self, logger=False):
        
        # Pretraining of CAE model
        cae = CAE(self._height, self._width, encoder_dim=self.cae_encoder_dim, lr=self.lr)
        
        trainer = pl.Trainer(devices=1, accelerator=self.lightning_device, max_epochs=self.pretraining_epochs, logger=logger)
        trainer.fit(cae, self.train_dataloader, self.val_dataloader)
        
        self.cae = cae
        
        # Training of full model
        self.clr = CLRmodel(height=self._height, width=self._width, num_cluster=self.num_cluster, 
                            ion_label_mat=self.ion_label_mat, activation=self.activation, encoder_dim=self.cae_encoder_dim, 
                            initial_upper=self.initial_upper, initial_lower=self.initial_lower, 
                            upper_iteration=self.upper_iteration, lower_iteration=self.lower_iteration,
                            dataset_specific_percentiles=self.dataset_specific_percentiles, lr=self.lr, 
                            cae_pretrained_model=cae, knn=self.KNN, knn_adj = self.knn_adj, 
                            cnn_dropout=self.cnn_dropout, weight_decay=self.weight_decay, clip_gradients=self.clip_gradients)
        
        trainer = pl.Trainer(devices=1, accelerator=self.lightning_device, max_epochs=self.training_epochs, logger=logger)
        trainer.fit(self.clr, self.train_dataloader, self.val_dataloader)
        
        return 0

    def inference_clusterlabels(self, new_data, cae=None, clr=None, normalize=True, device='cpu'):
        
        if cae is None:
            cae = self.cae   
        if clust is None:
            clust = self.clr
        
        with torch.no_grad():
            
            if normalize:
                new_data = self.image_normalization(new_data=new_data)
            
            test_x = torch.Tensor(nd, device=device)
            
            test_x = test_x.reshape((-1, 1, self._height, self._width))
            
            cae = cae.to(device)
            clust = clust.to(device)
            
            pseudo_label, x_p = clust(x_p)

            pseudo_label = torch.argmax(pseudo_label, dim=1)
            prediction_label.extend(pseudo_label.cpu().detach().numpy())
            prediction_label = np.array(prediction_label)

            return prediction_label

    def inference_embeddings(self, new_data, cae=None, clust=None, normalize_images=True, normalize_embeddings=True, device='cpu'):
        
        if cae is None:
            cae = self.cae   
        if clust is None:
            clust = self.clust
            
        with torch.no_grad():
            if normalize_images:
                new_data = self.image_normalization(new_data=new_data)
            
            test_x = torch.Tensor(nd, device=device)
            
            test_x = test_x.reshape((-1, 1, self._height, self._width))
            
            cae = cae.to(device)
            clust = clust.to(device)

            embeddings, x_p = clust(x_p)
            
            if normalize_embeddings:
                embeddings = functional.normalize(embeddings, p=2, dim=-1)
                embeddings = embeddings / embeddings.norm(dim=1)[:, None]

            return embeddings.cpu().detach().numpy()

    def inference_embeddings_train(self, device='cpu'):
        return self.inference_embeddings(new_data=self.data.train_dataset.images, normalize_images=False, normalize_embeddings=True, device=device)
    
    def inference_embeddings_val(self, device='cpu'):
        return self.inference_embeddings(new_data=self.data.val_dataset.images, normalize_images=False, normalize_embeddings=True, device=device)
    
    def inference_embeddings_test(self, device='cpu'):
        return self.inference_embeddings(new_data=self.data.test_dataset.images, normalize_images=False, normalize_embeddings=True, device=device)
