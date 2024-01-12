from platform import architecture
import torch
import numpy as np
from typing import Literal, Optional, Union, List

import torch.nn.functional as functional
import lightning.pytorch as pl

from .cae import CAE
from .selfContrastModel import selfContrastModel
from .colocContrastModel import colocContrastModel
from .regContrastModel import regContrastModel

from ...dataloader.IonImage_data import IonImagedata_random
from ...logger import DictLogger


class CRL:

    cae: CAE
    crl: Union[regContrastModel, colocContrastModel, selfContrastModel]

    def __init__(self,
                 data: IonImagedata_random,
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
                 loss_type: Literal['selfContrast', 'colocContrast', 'regContrast'] = 'selfContrast',
                 architecture: Literal['cnnclust', 'vit_b_16', 'resnet18', 'resnet34', 'resnet50', 
                                       'resnet101', 'resnet152'] = 'cnnclust',
                 resnet_pretrained: bool = False,
                 vitb16_pretrained: Optional[Literal['IMAGENET1K_V1', 'IMAGENET1K_SWAG_E2E_V1', 
                                                     'IMAGENET1K_SWAG_LINEAR_V1']] = None,
                 clip_gradients: Optional[float] = None,
                 overweight_cae: float = 1000,
                 cnn_dropout: float = 0.1,
                 weight_decay: float = 1e-4,
                 cae: bool = True):

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
        self.activation: Literal['softmax', 'relu', 'sigmoid'] = activation
        self.lr = lr
        self.pretraining_epochs = pretraining_epochs
        self.training_epochs = training_epochs
        self.mse_loss = torch.nn.MSELoss()
        self.cae_encoder_dim = cae_encoder_dim
        self.lightning_device = lightning_device
        self.clip_gradients = clip_gradients
        self.overweight_cae = overweight_cae
        self.cnn_dropout = cnn_dropout
        self.weight_decay = weight_decay

        self.loss_type: Literal['selfContrast', 'colocContrast', 'regContrast'] = loss_type
        if loss_type == 'selfContrast':
            self.model_cls = selfContrastModel
        elif loss_type == 'colocContrast':
            self.model_cls = colocContrastModel
        elif loss_type == 'regContrast':
            self.model_cls = regContrastModel
        else:
            raise ValueError('Loss type not available')

        print(f'After {self.training_epochs} epochs, the upper bound will be: '
              f'{self.initial_upper - (self.training_epochs * self.upper_iteration)}.')
        print(f'After {self.training_epochs} epochs, the lower bound will be: '
              f'{self.initial_lower + (self.training_epochs * self.lower_iteration)}.')

        if (self.initial_lower + (self.training_epochs * self.lower_iteration)) >= \
                (self.initial_upper - (self.training_epochs * self.upper_iteration)):
            raise ValueError(f'Lower percentile will be higher than upper percentile parameter '
                             f'after {self.training_epochs} epochs.\n'
                             f'Change initial_upper, initial_lower, '
                             f'upper_iteration, lower_iteration, '
                             f'or training_epochs parameters.')

        self.knn_adj = data.knn_adj            
        self.ion_label_mat = data.ion_label_mat

        self.train_dataloader = data.get_train_dataloader()
        
        self.val_dataloader = data.get_val_dataloader()

        self.use_cae = cae
    
        self.architecture: Literal['cnnclust', 'vit_b_16', 'resnet18', 'resnet34', 'resnet50', 
                                   'resnet101', 'resnet152'] = architecture
        
        self.resnet_pretrained = resnet_pretrained
        self.vitb16_pretrained: Optional[Literal['IMAGENET1K_V1', 'IMAGENET1K_SWAG_E2E_V1', 
                                                 'IMAGENET1K_SWAG_LINEAR_V1']] = vitb16_pretrained

    def image_normalization(self, new_data: np.ndarray):
        return self.data.image_normalization(new_data)

    def train(self, logger=False):
        
        if self.use_cae:
            # Pretraining of CAE model
            cae = CAE(self._height, self._width, encoder_dim=self.cae_encoder_dim, lr=self.lr)
            
            trainer = pl.Trainer(devices=1, accelerator=self.lightning_device, 
                                 max_epochs=self.pretraining_epochs, logger=logger, 
                                 callbacks=[]) # checkpoint_callback=False)
            trainer.fit(cae, self.train_dataloader, self.val_dataloader)
            
            self.cae = cae
        else:
            cae = False
        
        # Training of full model
        self.crl = self.model_cls(height=self._height, width=self._width, 
                                  num_cluster=self.num_cluster, 
                                  ion_label_mat=self.ion_label_mat, activation=self.activation, 
                                  encoder_dim=self.cae_encoder_dim, 
                                  initial_upper=self.initial_upper, 
                                  initial_lower=self.initial_lower, 
                                  upper_iteration=self.upper_iteration, 
                                  lower_iteration=self.lower_iteration,
                                  dataset_specific_percentiles=self.dataset_specific_percentiles, 
                                  lr=self.lr, 
                                  cae_pretrained_model=cae, knn=self.KNN, knn_adj = self.knn_adj, 
                                  cnn_dropout=self.cnn_dropout, weight_decay=self.weight_decay, 
                                  clip_gradients=self.clip_gradients, 
                                  architecture=self.architecture, 
                                  resnet_pretrained=self.resnet_pretrained, 
                                  vitb16_pretrained=self.vitb16_pretrained)
        
        dictlogger = DictLogger()
        trainer = pl.Trainer(devices=1, accelerator=self.lightning_device, 
                             max_epochs=self.training_epochs, logger=dictlogger)
        trainer.fit(self.crl, self.train_dataloader, self.val_dataloader)
        
        return dictlogger

    def inference_clusterlabels(self, new_data, crl=None, normalize=True, device='cpu'):
        
        if crl is None:
            crl = self.crl
        
        with torch.no_grad():
            
            if normalize:
                new_data = self.image_normalization(new_data=new_data)
            
            test_x = torch.tensor(new_data, device=device)
            
            test_x = test_x.reshape((-1, 1, self._height, self._width))
            
            crl = crl.to(device)
            
            pseudo_label, x_p = crl(test_x)

            pseudo_label = torch.argmax(pseudo_label, dim=1)
            prediction_label = list()
            prediction_label.extend(pseudo_label.cpu().detach().numpy())
            prediction_label = np.array(prediction_label)

            return prediction_label

    def inference_embeddings(self, new_data, crl=None, normalize_images=True, 
                             normalize_embeddings=True, device='cpu', use_embed_layer=False):
          
        if crl is None:
            crl = self.crl
            
        with torch.no_grad():
            if normalize_images:
                new_data = self.image_normalization(new_data=new_data)
            
            test_x = torch.tensor(new_data, device=device)
            
            test_x = test_x.reshape((-1, 1, self._height, self._width))
            
            crl = crl.to(device)
            crl.eval()

            if use_embed_layer:
                embeddings, _ = crl.embed_layers(test_x)
            else:
                embeddings, _ = crl(test_x)
            
            if normalize_embeddings:
                embeddings = functional.normalize(embeddings, p=2, dim=-1)
                embeddings = embeddings / embeddings.norm(dim=1)[:, None]

            return embeddings.cpu().detach().numpy()

    def inference_embeddings_train(self, device='cpu', use_embed_layer=False):
        return self.inference_embeddings(new_data=self.data.train_dataset.images, 
                                         normalize_images=False, 
                                         normalize_embeddings=True, device=device, 
                                         use_embed_layer=use_embed_layer)
    
    def inference_embeddings_val(self, device='cpu', use_embed_layer=False):
        return self.inference_embeddings(new_data=self.data.val_dataset.images, 
                                         normalize_images=False, 
                                         normalize_embeddings=True, device=device, 
                                         use_embed_layer=use_embed_layer)
    
    def inference_embeddings_test(self, device='cpu', use_embed_layer=False):
        return self.inference_embeddings(new_data=self.data.test_dataset.images, 
                                         normalize_images=False, 
                                         normalize_embeddings=True, device=device, 
                                         use_embed_layer=use_embed_layer)
    
    def get_loss_mask(self, latent: torch.Tensor, index, train_datasets, train_images, 
                      uu=90, ll=10, untransformed_images=None, device: str ='cpu'):
        
        if untransformed_images is None:
            tmp = self.crl.loss_mask(latent, uu, ll, train_datasets, index, train_images, 
                                    raw_images=train_images)
        else:
            tmp = self.crl.loss_mask(latent, uu, ll, train_datasets, index, train_images, 
                                    raw_images=untransformed_images)
        
        if self.loss_type == 'selfContrast':
            pos_loc, neg_loc, _ = tmp
            return (pos_loc - neg_loc).detach().cpu().numpy()
        
        elif self.loss_type == 'colocContrast':
            pos_loc, neg_loc, _ = tmp
            return (pos_loc - neg_loc).detach().cpu().numpy()
        
        elif self.loss_type == 'regContrast':
            ds_mask, _, _ = tmp

            return ds_mask.detach().cpu().numpy()
        
        else:
            raise ValueError('')
